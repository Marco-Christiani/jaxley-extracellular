"""Local typing helpers for interacting with partially typed ECS/Jaxley data."""

from __future__ import annotations

from typing import Any, TypedDict

from jax import Array


class AxialConductances(TypedDict):
    v: Array


class ECSParameters(TypedDict):
    axial_conductances: AxialConductances
    capacitance: Array
    area: Array


# data_stimulate returns a heterogeneous tuple consumed by jx.integrate.
DataStimuli = tuple[Any, ...]
