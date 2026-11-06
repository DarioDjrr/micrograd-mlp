"""Perceptrón multicapa (MLP) formado por capas densas con activación tanh."""

from Layer import Layer


class MLP:
    """Red neuronal feed-forward.

    Args:
        nInputs: número de entradas de la red.
        lenLayers: lista con el número de neuronas de cada capa; la última
            es la capa de salida. Ej.: ``[3, 4, 2]`` -> dos capas ocultas
            (3 y 4 neuronas) y 2 salidas.
    """

    def __init__(self, nInputs, lenLayers):
        sz = [nInputs] + lenLayers
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(lenLayers))]

    def __call__(self, inputs):
        """Propagación hacia delante; devuelve una lista de nodos de salida."""
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def parameters(self):
        """Devuelve todos los parámetros entrenables de la red."""
        return [p for layer in self.layers for p in layer.parameters()]
