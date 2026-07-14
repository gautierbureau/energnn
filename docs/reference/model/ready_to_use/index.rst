=================
Ready-to-use GNNs
=================

.. currentmodule:: energnn.model

We propose several ready-to-use implementations.
One just needs to pass an input structure and an output structure to the constructor.

All of them assemble the same pipeline
(:class:`~energnn.model.normalizer.TDigestNormalizer`, :class:`~energnn.model.encoder.MLPEncoder`,
:class:`~energnn.model.coupler.RecurrentCoupler`, :class:`~energnn.model.decoder.MLPEquivariantDecoder`)
and only differ by their capacity. Start small and scale up if the model underfits.

.. list-table::
    :header-rows: 1
    :widths: 26 16 16 20 16

    * - Model
      - Latent dim.
      - Steps ``N``
      - Hidden sizes
      - Breakpoints
    * - :class:`~energnn.model.ready_to_use.TinyRecurrentEquivariantGNN`
      - 4
      - 5
      - ``[]``
      - 10
    * - :class:`~energnn.model.ready_to_use.SmallRecurrentEquivariantGNN`
      - 8
      - 10
      - ``[16]``
      - 20
    * - :class:`~energnn.model.ready_to_use.MediumRecurrentEquivariantGNN`
      - 16
      - 20
      - ``[32]``
      - 50
    * - :class:`~energnn.model.ready_to_use.LargeRecurrentEquivariantGNN`
      - 32
      - 50
      - ``[64]``
      - 100
    * - :class:`~energnn.model.ready_to_use.ExtraLargeRecurrentEquivariantGNN`
      - 64
      - 200
      - ``[128, 128]``
      - 200

For full control over these hyper-parameters, use the configurable base class
:class:`~energnn.model.ready_to_use.ReadyRecurrentEquivariantGNN` directly.

.. autoclass:: energnn.model.ready_to_use.ReadyRecurrentEquivariantGNN
   :no-members:

.. autoclass:: energnn.model.ready_to_use.TinyRecurrentEquivariantGNN
   :no-members:

.. autoclass:: energnn.model.ready_to_use.SmallRecurrentEquivariantGNN
   :no-members:

.. autoclass:: energnn.model.ready_to_use.MediumRecurrentEquivariantGNN
   :no-members:

.. autoclass:: energnn.model.ready_to_use.LargeRecurrentEquivariantGNN
   :no-members:

.. autoclass:: energnn.model.ready_to_use.ExtraLargeRecurrentEquivariantGNN
   :no-members:
