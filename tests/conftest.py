import os

# must be set before importing jax, sensible cpu default
os.environ.setdefault("JAX_PLATFORMS", "cpu")
