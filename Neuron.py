"""Neurona individual: suma ponderada de las entradas + sesgo, seguida de tanh."""

import random

from nodo import nodo


class Neuron:
    """Neurona con ``nInputs`` pesos y un sesgo, inicializados en U(-1, 1).

    Calcula ``tanh(w1*x1 + ... + wn*xn + b)``.
    """

    def __init__(self, nInputs):
        self.weights = [nodo(random.uniform(-1, 1)) for _ in range(nInputs)]
        self.b = nodo(random.uniform(-1, 1))

    def __call__(self, inputs):
        """Evalúa la neurona sobre una lista de entradas (números o nodos)."""
        y = sum((xi * wi for xi, wi in zip(inputs, self.weights)), start=self.b)
        return nodo.tanh(y)

    def __repr__(self):
        pesos = ", ".join(f"w{i}={w.val:.3f}" for i, w in enumerate(self.weights))
        return f"Neuron({pesos}, b={self.b.val:.3f})"

    def parameters(self):
        """Devuelve los parámetros entrenables: pesos y sesgo."""
        return self.weights + [self.b]
