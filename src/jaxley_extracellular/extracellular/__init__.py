"""Phase-1 out-of-tree extracellular stimulation for Jaxley.

Public entry points
-------------------
- `discretization.build_voltage_operator_G` -- dense Jaxley-consistent G [1/ms]
- `field.point_source_potential`            -- phi_e(compartment, t) [mV]
- `equivalent_current.phi_e_to_ecs_nA`     -- phi_e -> i_ecs [nA]
- `jaxley_adapter.ensure_compartment_centers` -- populate x/y/z on module.nodes
- `jaxley_adapter.build_ecs_stimuli_nA`    -- full pipeline phi_e -> i_nA
"""
