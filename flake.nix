{
  description = "jaxley-extracellular";

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
        # Both jaxley and jaxley-mech ship a `tests/test_pickle.py` under
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
          jaxley-mech = prev."jaxley-mech".overrideAttrs (old: {
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
          jaxley-extracellular = ["gpu"];
        };

        devGpuDeps = {
          jaxley-extracellular = [
            "dev"
            "gpu"
          ];
        };

        devTpuDeps = {
          jaxley-extracellular = [
            "dev"
            "tpu"
          ];
        };

        testDeps = {
          jaxley-extracellular = ["dev"];
        };

        venv = pythonSetGpu.mkVirtualEnv "jaxley-extracellular" gpuDeps;
        devVenv = editablePythonSetGpu.mkVirtualEnv "jaxley-extracellular-dev" devGpuDeps;
        devTpuVenv = editablePythonSetTpu.mkVirtualEnv "jaxley-extracellular-dev-tpu" devTpuDeps;
        testVenv = pythonSetGpu.mkVirtualEnv "jaxley-extracellular-test" testDeps;

        cuda = pkgs.cudaPackages;

        # Google Cloud CLI (gcloud). Components must be selected via Nix.
        gcloud = pkgs.google-cloud-sdk;
        # if later want to talk to GKE clusters
        #   .withExtraComponents (with pkgs.google-cloud-sdk.components; [ gke-gcloud-auth-plugin ]);
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
            gcloud
          ];

          env = {
            UV_NO_SYNC = "1";
            UV_PYTHON = editablePythonSetTpu.python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";

            PYRIGHT_PYTHON_NODEJS_PATH = "${pkgs.nodejs_20}/bin/node";
          };

          shellHook = ''
            unset PYTHONPATH
            export REPO_ROOT="$(pwd -P)"
            echo "python: $(command -v python)"
            export TPU_LIBRARY_PATH="$(${devTpuVenv}/bin/python -c 'import pathlib, libtpu; print(pathlib.Path(libtpu.__file__).resolve().parent / "libtpu.so")' 2>/dev/null || true)"
            if [ -n "$TPU_LIBRARY_PATH" ] && [ -f "$TPU_LIBRARY_PATH" ]; then
              echo "TPU_LIBRARY_PATH=$TPU_LIBRARY_PATH"
            else
              unset TPU_LIBRARY_PATH
            fi
          '';
        };

        packages.default = venv;

        apps.default = {
          type = "app";
          program = "${venv}/bin/jaxley-extracellular";
          meta = {
            description = "jaxley-extracellular CLI";
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
