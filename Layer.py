"""Capa densa: conjunto de neuronas que reciben las mismas entradas."""

from Neuron import Neuron


class Layer:
    """Capa totalmente conectada de ``nNeurons`` neuronas con ``nInputs`` entradas."""

    def __init__(self, nInputs, nNeurons):
        self.neurons = [Neuron(nInputs) for _ in range(nNeurons)]

    def __call__(self, inputs):
        """Devuelve la lista con la salida de cada neurona."""
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        """Devuelve todos los parámetros de las neuronas de la capa."""
        return [p for neuron in self.neurons for p in neuron.parameters()]
