"""
Keras-Core (Keras 3) model scaffold using torch backend.
This file provides a minimal model and training loop using keras-core API.
Adjust the backend by setting environment variable KERAS_BACKEND to 'torch'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import keras
from keras import layers, models


def build_mlp(input_shape, hidden_units=(64, 32), activation='relu', output_units=1):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for u in hidden_units:
        x = layers.Dense(u, activation=activation)(x)
    outputs = layers.Dense(output_units)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def compile_and_summary(model, lr=1e-3):
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse')
    model.summary()


if __name__ == '__main__':
    # quick smoke test
    model = build_mlp((10,))
    compile_and_summary(model)
    x = np.random.randn(8, 10).astype(np.float32)
    y = np.random.randn(8, 1).astype(np.float32)
    model.fit(x, y, epochs=1, batch_size=4)
