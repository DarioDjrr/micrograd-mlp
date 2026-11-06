"""Entrenamiento de ejemplo de un MLP usando el motor de autodiferenciación.

Uso:
    python main.py
    python main.py --epochs 5000 --lr 0.05 --seed 42
"""

import argparse
import random

from nodo import nodo
from MLP import MLP

# Dataset de juguete: entradas de 4 valores -> 2 salidas en [-1, 1] (rango de tanh).
INPUTS = [
    [1, 2, 3, 4],
    [4, 3, 2, 1],
    [1, -1, 1, -1],
    [-1, -2, -3, -4],
]
TARGETS = [
    [1, -1],
    [-1, 1],
    [1, 1],
    [-1, -1],
]


def mse_loss(predicciones, objetivos):
    """Suma de errores cuadráticos entre las salidas de la red y los objetivos.

    Devuelve un ``nodo`` para poder retropropagar sobre él.
    """
    errores = [(p - nodo(t)) ** 2 for p, t in zip(predicciones, objetivos)]
    return sum(errores, start=nodo(0.0))


def train(mlp, inputs, targets, epochs, lr, log_every):
    """Entrena la red con descenso de gradiente (un paso por época, dataset completo).

    Devuelve la lista con la pérdida media de las épocas registradas.
    """
    historial = []
    for epoch in range(epochs):
        # 1. Poner a cero los gradientes acumulados
        for p in mlp.parameters():
            p.grad = 0.0

        # 2. Forward: pérdida total sobre todas las muestras
        perdida = sum(
            (mse_loss(mlp(x), y) for x, y in zip(inputs, targets)),
            start=nodo(0.0),
        )

        # 3. Backward: calcular gradientes
        nodo.backPropagation(perdida)

        # 4. Actualizar parámetros
        for p in mlp.parameters():
            p.val -= lr * p.grad

        if epoch % log_every == 0 or epoch == epochs - 1:
            historial.append(perdida.val)
            print(f"Época {epoch:>6} | Loss: {perdida.val:.6f}")
    return historial


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena un MLP desde cero.")
    parser.add_argument("--epochs", type=int, default=2000, help="número de épocas")
    parser.add_argument("--lr", type=float, default=0.05, help="tasa de aprendizaje")
    parser.add_argument("--hidden", type=int, nargs="+", default=[3, 4, 2],
                        help="neuronas por capa (la última es la salida)")
    parser.add_argument("--seed", type=int, default=None, help="semilla aleatoria")
    parser.add_argument("--log-every", type=int, default=100,
                        help="cada cuántas épocas mostrar la pérdida")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    if args.hidden[-1] != len(TARGETS[0]):
        raise SystemExit(f"La última capa debe tener {len(TARGETS[0])} neuronas.")

    mlp = MLP(len(INPUTS[0]), args.hidden)
    print(f"Red: {len(INPUTS[0])} -> {' -> '.join(map(str, args.hidden))} "
          f"({len(mlp.parameters())} parámetros)\n")

    train(mlp, INPUTS, TARGETS, args.epochs, args.lr, args.log_every)

    print("\nResultados finales:")
    for x, y in zip(INPUTS, TARGETS):
        salida = [round(o.val, 3) for o in mlp(x)]
        print(f"  entrada={x} -> salida={salida} (objetivo={y})")


if __name__ == "__main__":
    main()
