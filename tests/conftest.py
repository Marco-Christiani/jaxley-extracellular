import os

# must be set before importing jax, sensible cpu default
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# enable float64 for correctness tests
os.environ.setdefault("JAX_ENABLE_X64", "1")
