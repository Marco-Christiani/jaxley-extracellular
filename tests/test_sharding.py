"""Tests for sharding utilities (all run on CPU with 1 device)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxley_extracellular.extracellular.sharding import (
    config_sharding,
    iter_batches,
    make_device_mesh,
    pad_to_devices,
    shard_batch,
)

# ------------------------------------------------------------------
# make_device_mesh
# ------------------------------------------------------------------


class TestMakeDeviceMesh:
    def test_returns_mesh(self) -> None:
        mesh = make_device_mesh()
        assert isinstance(mesh, Mesh)

    def test_shape_matches_device_count(self) -> None:
        mesh = make_device_mesh()
        assert len(mesh.devices.flat) == jax.device_count()

    @pytest.mark.parametrize(
        ("axis_name", "expected"),
        [("d", ("d",)), ("batch", ("batch",))],
        ids=["default", "custom"],
    )
    def test_axis_name(self, axis_name: str, expected: tuple[str, ...]) -> None:
        mesh = make_device_mesh(axis_name=axis_name)
        assert mesh.axis_names == expected


# ------------------------------------------------------------------
# config_sharding
# ------------------------------------------------------------------


class TestConfigSharding:
    def test_returns_named_sharding(self) -> None:
        mesh = make_device_mesh()
        s = config_sharding(mesh)
        assert isinstance(s, NamedSharding)

    def test_partition_spec_uses_first_axis(self) -> None:
        mesh = make_device_mesh(axis_name="x")
        s = config_sharding(mesh)
        assert s.spec == P("x")  # type: ignore[no-untyped-call]


# ------------------------------------------------------------------
# pad_to_devices
# ------------------------------------------------------------------


class TestPadToDevices:
    @pytest.mark.parametrize(
        ("n_data", "n_devices", "expected_pad", "expected_shape"),
        [
            (4, 4, 0, (4,)),
            (5, 4, 3, (8,)),
            (3, 4, 1, (4,)),
        ],
        ids=["exact", "needs_padding", "one_short"],
    )
    def test_padding_count_and_shape(
        self, n_data: int, n_devices: int, expected_pad: int, expected_shape: tuple[int, ...]
    ) -> None:
        data = jnp.ones(n_data)
        padded, pad_count = pad_to_devices(data, n_devices)
        assert pad_count == expected_pad
        assert padded.shape == expected_shape

    def test_padding_values_are_zero(self) -> None:
        data = jnp.ones(3)
        padded, _ = pad_to_devices(data, 4)
        assert float(padded[3]) == 0.0

    def test_preserves_original_data(self) -> None:
        data = jnp.array([1.0, 2.0, 3.0])
        padded, _ = pad_to_devices(data, 2)
        assert jnp.allclose(padded[:3], data)

    def test_multidim(self) -> None:
        data = jnp.ones((3, 5))
        padded, pad_count = pad_to_devices(data, 2)
        assert padded.shape == (4, 5)
        assert pad_count == 1


# ------------------------------------------------------------------
# shard_batch
# ------------------------------------------------------------------


class TestShardBatch:
    def test_roundtrip_values(self) -> None:
        mesh = make_device_mesh()
        s = config_sharding(mesh)
        data = jnp.array([1.0, 2.0, 3.0])
        sharded = shard_batch(data, s)
        # On single device, padded length is a multiple of 1 -> same length
        assert jnp.allclose(sharded[:3], data)

    def test_output_is_jax_array(self) -> None:
        mesh = make_device_mesh()
        s = config_sharding(mesh)
        sharded = shard_batch(jnp.ones(4), s)
        assert isinstance(sharded, jax.Array)


# ------------------------------------------------------------------
# iter_batches
# ------------------------------------------------------------------


class TestIterBatches:
    def test_exact_division(self) -> None:
        data = jnp.arange(6)
        batches = list(iter_batches(data, batch_size=3))
        assert len(batches) == 2
        assert batches[0][0] == 0
        assert batches[1][0] == 3
        assert batches[0][1].shape == (3,)

    def test_remainder(self) -> None:
        data = jnp.arange(7)
        batches = list(iter_batches(data, batch_size=3))
        assert len(batches) == 3
        assert batches[-1][1].shape == (1,)

    def test_batch_larger_than_data(self) -> None:
        data = jnp.arange(3)
        batches = list(iter_batches(data, batch_size=100))
        assert len(batches) == 1
        assert batches[0][1].shape == (3,)


# ------------------------------------------------------------------
# Functional: jit(vmap) on sharded input
# ------------------------------------------------------------------


class TestFunctionalSharding:
    def test_jit_vmap_correct_results(self) -> None:
        mesh = make_device_mesh()
        s = config_sharding(mesh)
        data = jnp.array([1.0, 2.0, 3.0, 4.0])
        sharded = shard_batch(data, s)

        @jax.jit
        @jax.vmap
        def square(x: jax.Array) -> jax.Array:
            return x * x

        result = square(sharded)  # pyright: ignore[reportUnknownVariableType]
        expected = jnp.array([1.0, 4.0, 9.0, 16.0])
        assert jnp.allclose(result, expected)
