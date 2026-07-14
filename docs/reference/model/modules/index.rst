=======
Modules
=======


Components
----------

.. toctree::
    :maxdepth: 1

    normalizer/index
    encoder/index
    coupler/index
    decoder/index

Building block
--------------

Every trainable module is ultimately built from the same small multi-layer perceptron.
See :doc:`../mlp_catalog` for a catalog of where each MLP lives and what it means.

.. currentmodule:: energnn.model

.. autoclass:: MLP
   :show-inheritance:
