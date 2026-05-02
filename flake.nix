{
  description = "jaxley-extracellular";

  # Binary substituters for prebuilt store paths.
  # Users see a one-time `accept-flake-config` prompt unless their nix.conf already trusts these.
  # cuda-maintainers covers cudatoolkit/cudnn closures; nix-community covers many community packages.
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    uv2nix.url = "github:adisbladis/uv2nix";
    pyproject-nix.url = "github:nix-community/pyproject.nix";
    build-system-pkgs.url = "github:pyproject-nix/build-system-pkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    uv2nix,
    pyproject-nix,
    build-system-pkgs,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pythonGpu = pkgs.python311;
        pythonTpu = pythonGpu;

        # Load uv workspace (requires uv.lock + pyproject.toml).
        workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};

        pyproject-build-systems = import build-system-pkgs {
          inherit uv2nix pyproject-nix;
          inherit (pkgs) lib;
        };

        pythonBaseGpu = pkgs.callPackage pyproject-nix.build.packages {
          python = pythonGpu;
        };

        pythonBaseTpu = pkgs.callPackage pyproject-nix.build.packages {
          python = pythonTpu;
        };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Resolve file collisions in the virtualenv.
        # jaxley and jaxley-mech ship a `test_pickle.py` and
        # jaxley and kaleido ship a `conftest.py` under
        # site-packages/tests/, which pyproject.nix refuses to merge.
        # These tests are not needed at runtime, so we delete the `tests/` package
        # from jaxley-mech during install.
        # asciitree (zarr dep) ships only an sdist and needs setuptools.
        buildFixOverlay = final: prev: {
          asciitree = prev.asciitree.overrideAttrs (old: {
            nativeBuildInputs =
              (old.nativeBuildInputs or [])
              ++ [
                final.setuptools
              ];
          });
        };

        collisionFixOverlay = final: prev: {
          jaxley = prev."jaxley".overrideAttrs (old: {
            postInstall =
              (old.postInstall or "")
              + ''
                rm -rf $out/lib/python*/site-packages/tests
              '';
          });
        };

        pythonSetGpu = pythonBaseGpu.overrideScope (
          pkgs.lib.composeManyExtensions [
            pyproject-build-systems.wheel
            overlay
            buildFixOverlay
            collisionFixOverlay
          ]
        );

        pythonSetTpu = pythonBaseTpu.overrideScope (
          pkgs.lib.composeManyExtensions [
            pyproject-build-systems.wheel
            overlay
            buildFixOverlay
            collisionFixOverlay
          ]
        );

        # local dev setup
        editableOverlay = workspace.mkEditablePyprojectOverlay {
          root = "$REPO_ROOT";
          members = ["jaxley-extracellular"];
        };

        editablePythonSetGpu = pythonSetGpu.overrideScope editableOverlay;
        editablePythonSetTpu = pythonSetTpu.overrideScope editableOverlay;

        gpuDeps = {
          jaxley-extracellular = [
            "gpu"
            "tracking"
          ];
        };

        tpuDeps = {
          jaxley-extracellular = [
            "tpu"
            "tracking"
          ];
        };

        devGpuDeps = {
          jaxley-extracellular = [
            "dev"
            "gpu"
            "tracking"
          ];
        };

        devTpuDeps = {
          jaxley-extracellular = [
            "dev"
            "tpu"
            "tracking"
          ];
        };

        testDeps = {
          jaxley-extracellular = ["dev"];
        };

        gpuVenv = pythonSetGpu.mkVirtualEnv "jaxley-extracellular-gpu" gpuDeps;
        tpuVenv = pythonSetTpu.mkVirtualEnv "jaxley-extracellular-tpu" tpuDeps;
        devVenv = editablePythonSetGpu.mkVirtualEnv "jaxley-extracellular-dev" devGpuDeps;
        devTpuVenv = editablePythonSetTpu.mkVirtualEnv "jaxley-extracellular-dev-tpu" devTpuDeps;
        testVenv = pythonSetGpu.mkVirtualEnv "jaxley-extracellular-test" testDeps;

        # Bundle for Ray-on-TPU.
        #
        # Joins three things:
        #   - tpuVenv      : python + ray + jax + jaxley + ...
        #   - gperftools   : libtcmalloc.so for malloc replacement on the workload
        #   - rayHeadStart : the foreground head startup, baked into the closure
        #
        # The bundle's `bin/python` is overridden to set LD_PRELOAD then exec the
        # underlying interpreter. This matters for direct invocations (`nix run
        # .#tpu`, `~/jx-tpu-env/bin/python`). For Ray workers, env propagation
        # comes from the systemd unit's Environment block via fork inheritance,
        # not via the launcher. Single source of truth.
        tpuWorkerPython = pkgs.writeShellScript "python" ''
          export LD_PRELOAD="${pkgs.gperftools}/lib/libtcmalloc.so"
          # Persistent JAX compilation cache. Without this, every fresh process
          # pays the full XLA compile cost. The path is host-side (outside the
          # nix store) so it survives closure swaps. MIN_SIZE/MIN_COMPILE
          # overrides ensure even our small-but-slow-to-compile programs are
          # actually persisted (default thresholds skip them).
          export JAX_COMPILATION_CACHE_DIR="''${JAX_COMPILATION_CACHE_DIR:-$HOME/.jax_cache}"
          export JAX_COMPILATION_CACHE_MIN_SIZE_BYTES=0
          export JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
          exec ${tpuVenv}/bin/python "$@"
        '';

        # Foreground head startup. systemd's ExecStart points here. `--block`
        # keeps the ray processes attached to this script's lifetime so systemd
        # tracks the real head, not a fork that returns immediately.
        # Env vars (RAY_*) are read with safe defaults so the script also runs
        # standalone for debugging.
        #
        # LD_PRELOAD dance: the systemd unit exports LD_PRELOAD=...libtcmalloc
        # so the ray daemon (and forked workers) get tcmalloc. But that env
        # also poisons system tools we call during prelude (`rm` is system
        # /usr/bin/rm linked against host glibc 2.31, can't load nix glibc
        # 2.42's libtcmalloc). Cache the value, unset it for the prelude,
        # restore before exec so ray and its children still see it.
        rayHeadStart = pkgs.writeShellScriptBin "ray-head-start" ''
          set -euo pipefail
          saved_preload="''${LD_PRELOAD:-}"
          unset LD_PRELOAD
          rm -rf /tmp/ray
          export LD_PRELOAD="$saved_preload"
          exec ${tpuVenv}/bin/ray start --head \
            --node-ip-address=127.0.0.1 \
            --port=6379 \
            --ray-client-server-port="''${RAY_CLIENT_SERVER_PORT:-10001}" \
            --dashboard-host=0.0.0.0 \
            --dashboard-port="''${RAY_DASHBOARD_PORT:-8265}" \
            --metrics-export-port="''${RAY_METRICS_EXPORT_PORT:-8080}" \
            --num-gpus=0 \
            --resources="{\"TPU\": ''${RAY_TPU_RESOURCE_COUNT:-1}}" \
            --block
        '';

        tpuWorkerEnv = pkgs.symlinkJoin {
          name = "jaxley-extracellular-tpu-worker";
          paths = [tpuVenv pkgs.gperftools rayHeadStart];
          postBuild = ''
            rm $out/bin/python
            ln -sfn ${tpuWorkerPython} $out/bin/python
          '';
        };

        cuda = pkgs.cudaPackages;
        neuron = pkgs.neuron;
        neuronPython = pkgs.python313.withPackages (ps: [
          ps.numpy
        ]);
        neuronPythonPath = "${neuron}/lib/python3.13/site-packages";
        neuronRunner = pkgs.writeShellScriptBin "neuron-python" ''
          export PYTHONPATH="${neuronPythonPath}''${PYTHONPATH:+:$PYTHONPATH}"
          exec ${neuronPython}/bin/python "$@"
        '';

        # Google Cloud CLI (gcloud). Components must be selected via Nix.
        gcloud = pkgs.google-cloud-sdk;

        # Nix-built tcmalloc (Google's thread-caching malloc, ships in gperftools).
        # Used as LD_PRELOAD for the JAX/TPU workload to recover the perf the TPU
        # image's system tcmalloc preload was giving us. Sourcing it from nixpkgs
        # rather than /usr/lib/... keeps libunwind and libc in the nix store, so
        # nix-built binaries can resolve everything without leaking system paths.
        tcmallocLib = "${pkgs.gperftools}/lib/libtcmalloc.so";

        # Strip LD_PRELOAD from any nested `nix` invocation. Even with a nix-built
        # tcmalloc the daemon has been finicky historically, and there's no
        # benefit to preloading it for a build planner.
        nixWrapperFn = ''
          nix() { env -u LD_PRELOAD command nix "$@"; }
          export -f nix
        '';
        # if later want to talk to GKE clusters
        #   .withExtraComponents (with pkgs.google-cloud-sdk.components; [ gke-gcloud-auth-plugin ]);
        tex = pkgs.texlive.combine {
          inherit
            (pkgs.texlive)
            scheme-small # covers amsmath geometry fancyhdr hyperref graphicx xcolor inputenc etc.
            tcolorbox # tcolorbox with [many]
            tocloft
            biblatex
            ulem
            listings
            enumitem
            titlesec
            environ # tcolorbox dep
            tikzfill # tcolorbox breakable dep
            pdfcol # tcolorbox breakable dep
            bibtex # biblatex backend=bibtex
            latexmk # build automation
            cleveref # smart cross-references (\cref / \Cref)
            ;
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            devVenv
            pkgs.uv
            pkgs.nodejs_20
            pkgs.opentofu
            pkgs.go-task
            gcloud
            cuda.cudatoolkit
            cuda.cudnn
          ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = editablePythonSetGpu.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";

            # Use system Node.js on NixOS (avoid pyright-python nodeenv).
            PYRIGHT_PYTHON_NODEJS_PATH = "${pkgs.nodejs_20}/bin/node";
            XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cuda.cudatoolkit}";
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              cuda.cudatoolkit
              cuda.cudnn
              "/run/opengl-driver"
            ];
          };

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT="$(pwd -P)"
            echo "python: $(command -v python)"
          '';
        };

        devShells.tpu = pkgs.mkShell {
          packages = [
            devTpuVenv
            pkgs.uv
            pkgs.nodejs_20
            pkgs.opentofu
            pkgs.go-task
            pkgs.gperftools
            gcloud
          ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = editablePythonSetTpu.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";

            PYRIGHT_PYTHON_NODEJS_PATH = "${pkgs.nodejs_20}/bin/node";

            # Recover the JAX-on-TPU perf the system tcmalloc preload was buying us,
            # but from a nix-built lib so the loader never touches /usr/lib paths.
            LD_PRELOAD = tcmallocLib;
          };

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT="$(pwd -P)"
            echo "python: $(command -v python)"
            echo "LD_PRELOAD=$LD_PRELOAD"
            export TPU_LIBRARY_PATH="$(env -u LD_PRELOAD ${devTpuVenv}/bin/python -c 'import pathlib, libtpu; print(pathlib.Path(libtpu.__file__).resolve().parent / "libtpu.so")' 2>/dev/null || true)"
            if [ -n "$TPU_LIBRARY_PATH" ] && [ -f "$TPU_LIBRARY_PATH" ]; then
              echo "TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
            else
              unset TPU_LIBRARY_PATH
            fi
            ${nixWrapperFn}
          '';
        };

        devShells.neuron = pkgs.mkShell {
          packages = [
            neuron
            neuronPython
            neuronRunner
          ];

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT="$(pwd -P)"
            bbp_repo_dir="$REPO_ROOT/reference/bbp/simulation"
            if [ -d "$bbp_repo_dir" ]; then
              export BBP_SIM_DIR="$bbp_repo_dir"
              export BBP_MECH_SO="$bbp_repo_dir/x86_64/libnrnmech.so"
            else
              unset BBP_SIM_DIR
              unset BBP_MECH_SO
            fi
            echo "neuron-python: $(command -v neuron-python)"
            echo "nrnivmodl: $(command -v nrnivmodl)"
            if [ -n "$BBP_SIM_DIR" ]; then
              echo "BBP_SIM_DIR=$BBP_SIM_DIR"
            else
              echo "BBP_SIM_DIR is unset; copy vendored BBP assets into reference/bbp or set BBP_SIM_DIR manually"
            fi
          '';
        };

        devShells.paper = pkgs.mkShell {
          packages = [tex];
          shellHook = "exec ${pkgs.lib.getExe pkgs.fish}";
        };

        packages.gpu = gpuVenv;
        packages.tpu = tpuWorkerEnv;

        apps.gpu = {
          type = "app";
          program = "${gpuVenv}/bin/jaxley-extracellular";
          meta = {
            description = "jaxley-extracellular CLI (GPU venv)";
          };
        };

        apps.tpu = {
          type = "app";
          program = "${pkgs.writeShellScript "jaxley-extracellular-tpu" ''
            export LD_PRELOAD="${tcmallocLib}"
            exec "${tpuVenv}/bin/jaxley-extracellular" "$@"
          ''}";
          meta = {
            description = "jaxley-extracellular CLI (TPU venv, tcmalloc preload)";
          };
        };

        apps.neuron-python = {
          type = "app";
          program = "${neuronRunner}/bin/neuron-python";
          meta = {
            description = "Dedicated Python runner for NEURON-backed scripts";
          };
        };

        apps.paper = {
          type = "app";
          # Defaults to a one-shot pdf build of paper/paper.tex from the repo
          # root. Pass-through args let you do continuous preview / clean:
          #   nix run .#paper            # build
          #   nix run .#paper -- -pvc    # continuous mode (zathura)
          #   nix run .#paper -- -c      # clean aux files
          program = "${pkgs.writeShellScript "paper" ''
            set -euo pipefail
            PAPER_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/paper"
            cd "$PAPER_DIR"
            exec "${tex}/bin/latexmk" \
              -pdf \
              -interaction=nonstopmode \
              -e '$pdf_previewer = "zathura %O %S";' \
              "$@" paper.tex
          ''}";
          meta = {
            description = "Compile paper.";
          };
        };

        checks = let
          mkCheck = name: buildPhase:
            pkgs.stdenvNoCC.mkDerivation {
              inherit name buildPhase;
              src = ./.;
              nativeBuildInputs = [
                testVenv
                pkgs.nodejs_20
              ];
              installPhase = ''
                mkdir -p $out
              '';
            };
        in {
          pytest = mkCheck "jaxley-extracellular-pytest" ''
            export JAX_PLATFORMS=cpu
            ${testVenv}/bin/pytest -q
            echo "pytest done"
          '';

          ruff = mkCheck "jaxley-extracellular-ruff" ''
            ${testVenv}/bin/ruff check
            ${testVenv}/bin/ruff format --check
          '';

          mypy = mkCheck "jaxley-extracellular-mypy" ''
            ${testVenv}/bin/mypy
          '';

          pyright = mkCheck "jaxley-extracellular-pyright" ''
            export PYRIGHT_PYTHON_NODEJS_PATH="${pkgs.nodejs_20}/bin/node"
            ${testVenv}/bin/pyright --project pyproject.toml
          '';

          ty = mkCheck "jaxley-extracellular-ty" ''
            ${testVenv}/bin/ty check
          '';
        };
      }
    );
}
