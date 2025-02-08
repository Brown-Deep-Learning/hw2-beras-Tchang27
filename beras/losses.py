import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_total = np.mean(np.power(y_pred - y_true, 2), axis=-1)
        return np.mean(mse_total, axis=0)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        grad = 2 * (y_pred - y_true) / np.prod(y_pred.shape)
        return [Tensor(grad), Tensor(-grad)]

class CategoricalCrossEntropy(Loss):
    def clip(self, x, eps=1e-12):
        return np.clip(x, eps, 1-eps)
    
    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred = self.clip((y_pred))
        cce_total = -np.sum(y_true * np.log(y_pred), axis=1)
        return Tensor(np.mean(cce_total))

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        grad = y_pred - y_true
        lgrads = -np.log(self.clip(y_pred))
        return [Tensor(grad), Tensor(lgrads)]
        
