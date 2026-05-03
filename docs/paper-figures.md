# Reproducing Paper Figures

The paper figures are rendered from a pinned artifact bundle published on the
`paper-v0.1` GitHub release. The bundle contains figure inputs under
`results/paper_package/data/`, it does not rerun the TPU or NEURON workloads
that produced those inputs so it may be run without cloud infrastructure.

## Nix

Build the figures into a Nix store output:

```bash
nix build .#paper-figures
```

The generated PNG files are available through the `result` symlink.

For an in-place refresh of `paper/figures/`, run:

```bash
nix run .#paper-figures
```

By default this regenerates every figure. Pass a figure group to regenerate one
subset:

```bash
nix run .#paper-figures -- throughput
```

The Nix target fetches the release artifact by fixed hash and uses the same
Python dependency set as the development checks. Plotly/Kaleido render PNGs
through a Nix-provided Chromium executable via `BROWSER_PATH`.

## uv

Without Nix, use the locked Python environment and provide the artifact bundle
manually:

```bash
uv sync --frozen --group dev
mkdir -p results
tar -xzf paper-artifacts.tar.gz -C results
uv run python -m paper.make_figures --which all
```

The output files are written to `paper/figures/`.
