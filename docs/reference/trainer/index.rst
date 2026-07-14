=======
Trainer
=======

.. currentmodule:: energnn.trainer

The :class:`Trainer` orchestrates the learning loop: it iterates over a
:class:`~energnn.problem.ProblemLoader`, runs the model on each batch, back-propagates the objective
gradient returned by the problem, and updates the weights with an `Optax <https://optax.readthedocs.io/>`_
gradient transformation. Optionally it also evaluates on a validation loader, writes checkpoints, and logs
metrics to a :doc:`tracker <../tracker/index>`.

.. code-block:: python

    import optax
    from energnn.trainer import Trainer

    trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3))
    best_score = trainer.train(train_loader=train_loader, n_epochs=10)

At a glance:

- **Training** -- :meth:`Trainer.train` runs ``n_epochs`` over ``train_loader``. Each step calls the model,
  gets the gradient from the problem, and applies one optimizer update. It returns the best validation score.
- **Evaluation** -- pass a ``val_loader`` to :meth:`Trainer.train` (with ``eval_period`` / ``eval_after_epoch``),
  or call :meth:`Trainer.run_evaluation` directly to score the model over a loader.
- **Checkpointing** -- pass an `Orbax <https://orbax.readthedocs.io/>`_ ``CheckpointManager`` to persist the
  best model; restore it later with :meth:`Trainer.load_checkpoint`.
- **Tracking** -- pass a :class:`~energnn.tracker.Tracker` to stream scores and info dictionaries to an
  experiment backend.

Set ``optim_mode`` to ``"minimize"`` (default) or ``"maximize"`` depending on whether a lower or higher
score is better; it drives both early-stopping bookkeeping and which checkpoint is kept as "best".

.. autoclass:: Trainer
   :no-members:

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Trainer.train
   Trainer.run_evaluation
   Trainer.save_checkpoint
   Trainer.load_checkpoint
