"""Device mesh and batch distribution utilities for multi-device sweeps.

Thin wrappers over JAX's sharding API. On single-device (CPU/single GPU),
everything degrades gracefully to no-ops (mesh has shape ``(1,)``).
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array


def make_device_mesh(axis_name: str = "d") -> Mesh:
    """Create a 1-D device mesh spanning all available devices."""
    return Mesh(jax.devices(), axis_names=(axis_name,))  # pyright: ignore[reportUnknownVariableType]


def config_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding that partitions dimension 0 across the mesh."""
    return NamedSharding(mesh, P(mesh.axis_names[0]))  # type: ignore[no-untyped-call]


def pad_to_devices(data: Array, n_devices: int) -> tuple[Array, int]:
    """Pad leading dim to a multiple of *n_devices*.

    Returns ``(padded_array, pad_count)``; caller trims output by *pad_count*.
    """
    n = data.shape[0]
    remainder = n % n_devices
    if remainder == 0:
        return data, 0
    pad_count = n_devices - remainder
    pad_shape = (pad_count, *data.shape[1:])
    padding = jnp.zeros(pad_shape, dtype=data.dtype)
    return jnp.concatenate([data, padding], axis=0), pad_count


def shard_batch(data: Array, sharding: NamedSharding) -> Array:
    """Pad and place *data* across devices according to *sharding*.

    The leading dimension is padded to a multiple of the device count so that
    each device receives an equal slice.
    """
    n_devices = jax.device_count()
    padded, _ = pad_to_devices(data, n_devices)
    return jax.device_put(padded, sharding)  # type: ignore[no-any-return]


def iter_batches(data: Array, batch_size: int) -> Iterator[tuple[int, Array]]:
    """Yield ``(start_idx, chunk)`` for sequential batch processing."""
    n = data.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, data[start:end]
