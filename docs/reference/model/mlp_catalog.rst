===============
Catalog of MLPs
===============

.. currentmodule:: energnn.model

Every trainable weight in an **EnerGNN** model lives inside a small
:class:`~energnn.model.MLP`. A GNN is not one large network but a *family* of these
little MLPs, each attached to a specific object class or port and **reused** across every
object (or address) of that kind. This page lists them all: their symbol, where they live
in the code, how many are instantiated, what they map, and what they mean.

For a visual, per-stage placement of these MLPs on a toy graph, see the
"Where do the trainable weights live?" figure in :doc:`the basics </basics>`.

Notation
========

- :math:`c` -- an object class (a key of the graph structure, e.g. ``"lines"``).
- :math:`o` -- a port of a class (e.g. ``"bus1"``).
- :math:`x_e` -- the raw features of hyper-edge :math:`e`; :math:`z_e` its encoded features.
- :math:`h_a` -- the latent coordinate carried by address :math:`a`; :math:`d` the latent dimension.
- :math:`x_e := (x_e)` and :math:`h_e := (h_{o(e)})_{o \in \mathcal{O}^c}` denote, respectively, the features of an
  edge and the concatenation of the coordinates sitting at its ports.

Summary
=======

.. list-table::
    :header-rows: 1
    :widths: 16 24 18 22 20

    * - Symbol
      - Lives in
      - How many
      - Maps
      - Role
    * - :math:`\phi_c` (encoder)
      - :class:`~energnn.model.encoder.MLPEncoder` ``.mlp_dict[c]``
      - one per context class **with features**
      - features :math:`x_e` → latent :math:`z_e \in \mathbb{R}^d`
      - Embed each object's own features.
    * - :math:`\xi_{c,o}` (message)
      - :class:`~energnn.model.coupler.LocalSumMessagePassingFunction` ``.mlp_tree[c][o]``
      - one per **(class, port)**
      - :math:`[\,h_e,\, z_e\,]` → message :math:`\in \mathbb{R}^d`
      - Local message from an edge onto port :math:`o`'s address.
    * - :math:`\phi` (coupler)
      - :class:`~energnn.model.coupler.RecurrentCoupler` / :class:`~energnn.model.coupler.NODECoupler` ``.phi``
      - one, **shared** by all addresses
      - concat. of messages → update :math:`\in \mathbb{R}^d`
      - Turn messages into the (ODE) update direction.
    * - :math:`\phi_c` (decoder)
      - :class:`~energnn.model.decoder.MLPEquivariantDecoder` ``.mlp_dict[c]``
      - one per **output** class
      - :math:`[\,h_e,\, z_e\,]` → decision features
      - Read coordinates back into a per-object decision.
    * - :math:`\psi,\ \phi` (decoder)
      - :class:`~energnn.model.decoder.SumInvariantDecoder` / :class:`~energnn.model.decoder.MeanInvariantDecoder`
      - one each, **global**
      - :math:`h_a` → ... → global vector
      - Aggregate all addresses into one output vector.

.. note::

    The **normalizer** (:class:`~energnn.model.normalizer.TDigestNormalizer`,
    :class:`~energnn.model.normalizer.CenterReduceNormalizer`) holds **no MLP**: it only tracks feature
    statistics used to rescale inputs. All learnable weights sit in the encoder, coupler and decoder.

The building block: ``MLP``
===========================

All the entries below are instances of the same class, :class:`~energnn.model.MLP`
(``src/energnn/model/utils.py``) -- a plain multi-layer perceptron built from ``flax.nnx.Linear``
layers. Everything specific to a model (how many MLPs, their input/output sizes, where they are applied)
is decided by the modules that instantiate them, described next.

Encoder MLPs -- :math:`\phi_c`
==============================

.. math::

    \forall c \in \mathcal{C},\ \forall e \in \mathcal{E}^c_x,\quad z_e = \phi_\theta^c(x_e).

- **Code:** :class:`~energnn.model.encoder.MLPEncoder`, attribute ``mlp_dict`` (built by ``_build_mlp_dict``),
  in ``src/energnn/model/encoder/mlp_encoder.py``.
- **Count:** one MLP per context class that has at least one feature. Classes without features map to ``None``.
- **Shape:** input = number of features of class :math:`c`; output = latent dimension :math:`d`.
- **Meaning:** lift each object's raw features into the shared latent space, independently of the graph structure.

Message-passing MLPs -- :math:`\xi_{c,o}`
=========================================

.. math::

    \psi_\theta(h,x)_a = \sigma\!\left(\sum_{(c,e,o)\in \mathcal{N}_x(a)} \xi^{c,o}_\theta(h_e, x_e)\right).

- **Code:** :class:`~energnn.model.coupler.LocalSumMessagePassingFunction`, attribute ``mlp_tree`` (built by
  ``_build_mlp_tree``), in ``src/energnn/model/coupler/message_passing/message_passing_function.py``.
  ``mlp_tree`` is a nested dict ``{class: {port: MLP}}``.
- **Count:** one MLP per ``(class, port)`` pair (ports may be excluded via ``port_scatter_blacklist``).
- **Shape:** input = :math:`d \times (\text{number of ports of } c)` for the port coordinates,
  plus the encoded feature size (or the raw feature count if the graph is not encoded); output = :math:`d`.
- **Meaning:** every port of an edge produces a message from the edge's context :math:`[h_e, z_e]`; the message is
  **scatter-added** onto that port's address, then the outer activation :math:`\sigma` is applied to the sum.
- **Variant:** :class:`~energnn.model.coupler.IdentityMessagePassingFunction` holds **no MLP** -- it returns the
  coordinates unchanged, which is useful as a residual/skip term.

Coupler outer MLP -- :math:`\phi`
=================================

.. math::

    h_a \leftarrow h_a + \Delta t \cdot \phi_\theta\big(\psi^1_\theta(h;x)_a, \dots, \psi^n_\theta(h;x)_a\big).

- **Code:** attribute ``phi`` of :class:`~energnn.model.coupler.RecurrentCoupler` (explicit Euler steps) or
  :class:`~energnn.model.coupler.NODECoupler` (adaptive ODE solver via *Diffrax*), in
  ``src/energnn/model/coupler/message_passing/``.
- **Count:** a single MLP, **shared** across every address and every message-passing step.
- **Shape:** input = the concatenation of the outputs of all message functions
  (:math:`n \times d` for :math:`n` message functions); output = :math:`d`.
- **Meaning:** combine the per-address messages into the update direction of the coordinates. Applying the *same*
  :math:`\phi` at every address and every step is what keeps the coupler a compact, size-independent operator.

Equivariant decoder MLPs -- :math:`\phi_c`
==========================================

.. math::

    \forall c \in \mathcal{C},\ \forall e \in \mathcal{E}^c_x,\quad \hat{y}_e = \phi_\theta^c(x_e, h_e).

- **Code:** :class:`~energnn.model.decoder.MLPEquivariantDecoder`, attribute ``mlp_dict`` (built by
  ``_build_mlp_dict``), in ``src/energnn/model/decoder/equivariant_decoder.py``.
- **Count:** one MLP per **output** class (every output class must also appear in the context).
- **Shape:** input = :math:`d \times (\text{number of ports of } c)` plus the encoded feature size (or raw feature
  count); output = the number of decision features of class :math:`c`.
- **Meaning:** for each object, combine the coordinates gathered at its ports with its context features to produce
  the object's decision. Because it is applied per object, the output is permutation-**equivariant**.

Invariant decoder MLPs -- :math:`\psi,\ \phi`
=============================================

.. math::

    \hat{y} = \phi_\theta\!\left(\tfrac{1}{|\mathcal{A}(x)|}\sum_{a}\psi_\theta(h_a)\right)
    \quad\text{or}\quad
    \hat{y} = \phi_\theta\!\left(\sum_{a}\psi_\theta(h_a)\right).

- **Code:** :class:`~energnn.model.decoder.SumInvariantDecoder` and
  :class:`~energnn.model.decoder.MeanInvariantDecoder`, attributes ``psi`` (inner) and ``phi`` (outer), in
  ``src/energnn/model/decoder/invariant_decoder.py``.
- **Count:** one inner and one outer MLP, both **global**.
- **Shape:** ``psi`` maps :math:`d` → :math:`d'`; ``phi`` maps :math:`d'` → the global output dimension.
- **Meaning:** aggregate the coordinates of *all* addresses into a single, order-independent (permutation-**invariant**)
  output vector. Use this instead of the equivariant decoder when the target is one global quantity rather than a
  per-object decision.

Inspecting the MLPs at runtime
==============================

Because each MLP is a plain attribute, you can walk the whole family on any instantiated model:

.. code-block:: python

    from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN

    model = TinyRecurrentEquivariantGNN(in_structure=..., out_structure=...)

    model.encoder.mlp_dict                        # {"lines": MLP, "generators": MLP, "loads": MLP, ...}
    model.coupler.message_functions[0].mlp_tree   # {"lines": {"bus1": MLP, "bus2": MLP}, ...}
    model.coupler.phi                             # the single shared outer MLP
    model.decoder.mlp_dict                        # {"switches": MLP, "generators": MLP}

    # Each leaf is an energnn.model.utils.MLP; inspect its shape via
    model.encoder.mlp_dict["generators"].in_size, model.encoder.mlp_dict["generators"].out_size

See also
========

- :doc:`the basics </basics>` for the pipeline and the weight-map figure.
- :doc:`modules/index` for the full API of the encoder, coupler, decoder and normalizer modules.
- :doc:`ready_to_use/index` for pre-assembled models and their exact sizes.
