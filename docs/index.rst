jaxley-extracellular
====================

**Differentiable extracellular stimulation - from electrode waveform to membrane response.**

``jaxley-extracellular`` extends `Jaxley <https://jaxley.readthedocs.io>`_ with a full
extracellular stimulation pipeline. Every step is JAX-traceable: differentiate through
electrode position, waveform amplitude, tissue conductivity, and HH integration.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   getting-started
   paper-figures

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Concepts

   concepts/pipeline
   concepts/waveforms

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api

What it does
------------

Given a current waveform and electrode position, the pipeline:

1. Computes extracellular potential ``phi_e`` at each compartment center (point-source model)
2. Builds the voltage diffusion operator ``G`` consistent with Jaxley's internal solver
3. Converts ``phi_e`` into equivalent injected current via ``G @ phi_e``
4. Integrates with ``jx.integrate`` using Jaxley's standard ``data_stimuli`` API

Why it matters
--------------

The implementation keeps the extracellular stimulation path in JAX arrays.
That means waveform amplitudes, electrode positions, conductivity, and solver
outputs can participate in ``jax.jit``, ``jax.vmap``, and ``jax.grad`` workflows.
The lower-level modules expose each physical conversion step directly; the
:class:`~jaxley_extracellular.extracellular.experiment.ECSExperiment` wrapper
caches the static geometry and operator pieces for batched sweeps.

Install
-------

.. code-block:: bash

   pip install jaxley-extracellular

Or with Nix:

.. code-block:: bash

   nix develop        # GPU shell
   nix develop .#tpu  # TPU shell

Build the docs
--------------

.. code-block:: bash

   nix build .#docs   # static site derivation
   nix run .#docs     # build and serve locally on http://localhost:8001

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
