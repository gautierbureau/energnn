=======
Model
=======

.. currentmodule:: energnn.model

Our core GNN implementations have been designed to process H2MG data.
A :class:`~energnn.model.GNN` chains four swappable modules -- a normalizer, an encoder, a coupler and a
decoder -- as sketched below (see :doc:`the basics </basics>` for a walk-through).

.. image:: /_static/energnn_gnn_pipeline_black.svg
    :class: only-light
    :align: center
    :width: 100%
    :alt: The GNN forward pass chains a normalizer, an encoder, a coupler and a decoder.

.. image:: /_static/energnn_gnn_pipeline_white.svg
    :class: only-dark
    :align: center
    :width: 100%
    :alt: The GNN forward pass chains a normalizer, an encoder, a coupler and a decoder.

There are 3 ways of constructing a GNN.

- Build a :class:`~energnn.model.GNN` by combining the various module implementations provided in **EnerGNN**.
- Build a :class:`~energnn.model.GNN` by combining your own module implementations with the ones
  that are pre-existing in **EnerGNN** (just make sure to respect interfaces).
- Use a **ready-to-use** GNN implementation provided in the :mod:`energnn.model.ready_to_use` module.

Components
----------

.. toctree::
    :maxdepth: 1

    core_gnn/index
    modules/index
    ready_to_use/index
