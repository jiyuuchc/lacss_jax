from functools import partial

import flax.linen as nn
import jax
import optax
import pytest

import lacss.train


def gen(X, Y):
    for x, y in zip(X, Y):
        yield x, y


class Mse(lacss.train.Loss):
    def call(self, preds, target, **kwargs):
        return ((preds - target) ** 2).mean()


def test_eager_strategy():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 1, 16])
    _Y = jax.random.uniform(k2, [16, 1, 4])

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=Mse(),
        train_strategy=lacss.train.strategy.Eager,
        optimizer=optax.adam(0.01),
    )

    g = partial(gen, X=_X, Y=_Y)

    trainer.initialize(g)

    for epoch in range(10):
        for log in trainer.train(g):
            pass

    assert log["mse"] < 0.05


def test_vmap_strategy():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    _X = jax.random.uniform(k1, [16, 1, 16])
    _Y = jax.random.uniform(k2, [16, 1, 4])

    trainer = lacss.train.Trainer(
        model=nn.Dense(4),
        losses=Mse(),
        train_strategy=lacss.train.strategy.VMapped,
        optimizer=optax.adam(0.01),
    )

    g = partial(gen, X=_X, Y=_Y)

    trainer.initialize(g)

    for epoch in range(10):
        for log in trainer.train(g):
            pass

    assert log["mse"] < 0.05
