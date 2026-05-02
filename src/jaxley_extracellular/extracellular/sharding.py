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
    """Create a 1-D device mesh spanning all available devices.

    Parameters
    ----------
    axis_name : str, optional
        Name of the single mesh axis (default ``"d"``).

    Returns
    -------
    jax.sharding.Mesh
        1-D mesh of shape ``(n_devices,)``.
    """
    return Mesh(jax.devices(), axis_names=(axis_name,))  # pyright: ignore[reportUnknownVariableType]


def config_sharding(mesh: Mesh) -> NamedSharding:
    """Build a sharding that partitions dimension 0 across the mesh.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        The device mesh to shard over.

    Returns
    -------
    jax.sharding.NamedSharding
        Sharding with leading-axis partitioned over ``mesh``.
    """
    return NamedSharding(mesh, P(mesh.axis_names[0]))  # type: ignore[no-untyped-call]


def pad_to_devices(data: Array, n_devices: int) -> tuple[Array, int]:
    """Pad leading dim to a multiple of ``n_devices``.

    Parameters
    ----------
    data : Array
        Input array. The leading axis is padded with zeros if needed.
    n_devices : int
        Number of devices the leading axis should divide evenly across.

    Returns
    -------
    padded : Array
        Array with leading axis a multiple of ``n_devices``.
    pad_count : int
        Number of zero rows appended. Callers should slice the output
        back to the original leading length using this count.
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
    """Pad and place ``data`` across devices according to ``sharding``.

    The leading dimension is padded to a multiple of the device count
    so that each device receives an equal slice.

    Parameters
    ----------
    data : Array
        Batch with leading axis to shard.
    sharding : jax.sharding.NamedSharding
        Target sharding (e.g.\ from :func:`config_sharding`).

    Returns
    -------
    Array
        Sharded array placed on devices.
    """
    n_devices = jax.device_count()
    padded, _ = pad_to_devices(data, n_devices)
    return jax.device_put(padded, sharding)  # type: ignore[no-any-return]


def iter_batches(data: Array, batch_size: int) -> Iterator[tuple[int, Array]]:
    """Yield ``(start_idx, chunk)`` pairs for sequential batch processing.

    Parameters
    ----------
    data : Array
        Input array; the leading axis is the batch axis.
    batch_size : int
        Maximum length of each yielded chunk.

    Yields
    ------
    start : int
        Starting index of the chunk along the leading axis.
    chunk : Array
        Slice of ``data`` of length up to ``batch_size``.
    """
    n = data.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, data[start:end]
