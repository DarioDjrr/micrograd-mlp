# micrograd-mlp

Perceptrón multicapa (MLP) construido desde cero en Python puro, con su propio motor de autodiferenciación. Sin NumPy ni PyTorch: solo la librería estándar. Proyecto educativo para familiarizarme con la retropropagación (backpropagation).

## Características

- Motor de autodiferenciación en modo inverso para escalares (`nodo`).
- Operaciones soportadas: `+`, `-`, `*`, `/`, `**` y `tanh`, también con escalares a la izquierda (`2 * x`).
- Clases `Neuron`, `Layer` y `MLP` para construir redes de cualquier tamaño.
- Entrenamiento con descenso de gradiente y pérdida cuadrática.
- Tests que verifican los gradientes frente a derivadas numéricas.

## Estructura

```
├── nodo.py        # Motor de autodiferenciación (grafo computacional + backprop)
├── Neuron.py      # Neurona: tanh(w·x + b)
├── Layer.py       # Capa densa de neuronas
├── MLP.py         # Perceptrón multicapa
├── main.py        # Script de entrenamiento de ejemplo (con CLI)
├── test_nodo.py   # Tests unitarios
└── LICENSE
```

## Requisitos

Python 3.8 o superior. No hay dependencias externas.

## Uso

```bash
git clone https://github.com/DarioDjrr/micrograd-mlp.git
cd micrograd-mlp
python main.py
```

Opciones:

| Argumento     | Descripción                                     | Por defecto |
|---------------|-------------------------------------------------|-------------|
| `--epochs`    | Número de épocas de entrenamiento               | `2000`      |
| `--lr`        | Tasa de aprendizaje                             | `0.05`      |
| `--hidden`    | Neuronas por capa (la última es la de salida)   | `3 4 2`     |
| `--seed`      | Semilla aleatoria para reproducibilidad         | aleatoria   |
| `--log-every` | Cada cuántas épocas imprimir la pérdida         | `100`       |

Ejemplo:

```bash
python main.py --epochs 3000 --lr 0.03 --hidden 5 5 2 --seed 42
```

Salida típica:

```
Red: 4 -> 3 -> 4 -> 2 (41 parámetros)

Época      0 | Loss: 14.934312
...
Resultados finales:
  entrada=[1, 2, 3, 4] -> salida=[0.989, -0.988] (objetivo=[1, -1])
```

## Uso como librería

```python
from nodo import nodo
from MLP import MLP

# Autodiferenciación
a, b = nodo(2.0), nodo(3.0)
c = (a * b + 1).tanh()
nodo.backPropagation(c)
print(a.grad, b.grad)

# Red neuronal: 4 entradas, capas de 3, 4 y 2 neuronas
red = MLP(4, [3, 4, 2])
salida = red([1, 2, 3, 4])   # lista de 2 nodos
```

## Cómo funciona

1. **Forward**: cada operación crea un `nodo` que guarda su valor, sus operandos y una función `backward`
   con la derivada local.
2. **Backward**: `nodo.backPropagation` ordena el grafo topológicamente y lo recorre al revés aplicando la
   regla de la cadena; los gradientes se acumulan en `grad`.
3. **Actualización**: `p.val -= lr * p.grad` para cada parámetro de la red.

> Importante: los gradientes se acumulan, así que hay que ponerlos a `0.0` antes de cada pasada (lo hace `train()`).

## Tests

```bash
python -m unittest -v
```

## Licencia

MIT, ver [LICENSE](LICENSE).
