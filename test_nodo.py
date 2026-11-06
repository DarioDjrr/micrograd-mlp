"""Tests del motor de autodiferenciación (comparan con derivadas numéricas).

Ejecutar con:  python -m unittest -v
"""

import math
import unittest

from nodo import nodo


def grad_numerico(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


class TestNodo(unittest.TestCase):
    def test_suma_y_producto(self):
        a, b = nodo(2.0), nodo(3.0)
        c = a * b + 1
        nodo.backPropagation(c)
        self.assertEqual(c.val, 7.0)
        self.assertEqual((a.grad, b.grad), (3.0, 2.0))

    def test_resta_y_division(self):
        a, b = nodo(6.0), nodo(2.0)
        c = (a - b) / b
        nodo.backPropagation(c)
        self.assertAlmostEqual(c.val, 2.0)
        self.assertAlmostEqual(a.grad, 0.5)
        self.assertAlmostEqual(b.grad, -1.5)  # d/db[(a-b)/b] = -a/b²

    def test_tanh(self):
        x = nodo(0.5)
        y = x.tanh()
        nodo.backPropagation(y)
        esperado = grad_numerico(math.tanh, 0.5)
        self.assertAlmostEqual(x.grad, esperado, places=5)

    def test_potencia(self):
        x = nodo(3.0)
        y = x ** 3
        nodo.backPropagation(y)
        self.assertAlmostEqual(x.grad, 27.0)

    def test_escalar_a_la_izquierda(self):
        x = nodo(4.0)
        y = 10 - 2 * x + 8 / x
        nodo.backPropagation(y)
        self.assertAlmostEqual(y.val, 4.0)
        self.assertAlmostEqual(x.grad, -2 - 8 / 16)

    def test_nodo_reutilizado(self):
        x = nodo(3.0)
        y = x * x
        nodo.backPropagation(y)
        self.assertAlmostEqual(x.grad, 6.0)


if __name__ == "__main__":
    unittest.main()
