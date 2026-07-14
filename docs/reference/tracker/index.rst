=======
Tracker
=======

.. currentmodule:: energnn.tracker

A **tracker** monitors a training run: it records the configuration, logs the scores and info
dictionaries produced by the :class:`~energnn.trainer.Trainer`, and can reference datasets or
artifacts. Pass a tracker to :meth:`~energnn.trainer.Trainer.train` to enable monitoring.

Implement the :class:`Tracker` interface to plug in your own backend, or use the provided
:class:`MlflowTracker`.

Interface
=========

.. autoclass:: Tracker
   :members:
   :show-inheritance:

Implementations
===============

.. autoclass:: MlflowTracker
   :members:
   :show-inheritance:
