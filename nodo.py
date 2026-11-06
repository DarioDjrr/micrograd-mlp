"""Motor de autodiferenciación en modo inverso (reverse-mode autodiff).

Define la clase :class:`nodo`, que envuelve un valor escalar y recuerda cómo
se calculó, de forma que se pueda obtener el gradiente de cualquier resultado
respecto a todas sus entradas mediante retropropagación (backpropagation).
"""

import math


class nodo:
    """Nodo de un grafo computacional con soporte de gradientes.

    Atributos:
        val: valor numérico del nodo.
        hijos: nodos de los que depende este nodo (sus operandos).
        op: símbolo de la operación que lo generó (solo informativo).
        grad: derivada de la salida final respecto a este nodo.
        et: etiqueta opcional para depurar.
        backward: función que propaga ``grad`` hacia los hijos (regla de la cadena).

    Ejemplo:
        >>> a, b = nodo(2.0), nodo(3.0)
        >>> c = a * b + 1
        >>> nodo.backPropagation(c)
        >>> a.grad, b.grad
        (3.0, 2.0)
    """

    def __init__(self, val, hijos=(), op='', et=''):
        self.val = val
        self.hijos = hijos
        self.op = op
        self.grad = 0.0
        self.et = et
        self.backward = None

    def __repr__(self):
        return (f"nodo({self.et} = {self.val}. Grad = {self.grad}, "
                f"op = {self.op}, hijos = {self.hijos})")

    @staticmethod
    def isScalar(other):
        """Devuelve True si ``other`` es un ``int`` o ``float`` de Python."""
        return isinstance(other, (int, float))

    # ------------------------------------------------------------------
    # Operaciones básicas (nodo <op> nodo/escalar)
    # ------------------------------------------------------------------
    def __add__(self, other):
        """Suma: d(a+b)/da = 1, d(a+b)/db = 1."""
        es_nodo = not nodo.isScalar(other)
        if es_nodo:
            res = nodo(self.val + other.val, [self, other], '+')
        else:
            res = nodo(self.val + other, [self], '+')

        def backward():
            self.grad += res.grad
            if es_nodo:
                other.grad += res.grad
        res.backward = backward
        return res

    def __sub__(self, other):
        """Resta: d(a-b)/da = 1, d(a-b)/db = -1."""
        es_nodo = not nodo.isScalar(other)
        if es_nodo:
            res = nodo(self.val - other.val, [self, other], '-')
        else:
            res = nodo(self.val - other, [self], '-')

        def backward():
            self.grad += res.grad
            if es_nodo:
                other.grad -= res.grad
        res.backward = backward
        return res

    def __mul__(self, other):
        """Multiplicación: d(a*b)/da = b, d(a*b)/db = a."""
        es_nodo = not nodo.isScalar(other)
        if es_nodo:
            res = nodo(self.val * other.val, [self, other], '*')
        else:
            res = nodo(self.val * other, [self], '*')

        def backward():
            if es_nodo:
                self.grad += res.grad * other.val
                other.grad += res.grad * self.val
            else:
                self.grad += res.grad * other
        res.backward = backward
        return res

    def __truediv__(self, other):
        """División: d(a/b)/da = 1/b, d(a/b)/db = -a/b²."""
        es_nodo = not nodo.isScalar(other)
        if es_nodo:
            res = nodo(self.val / other.val, [self, other], '/')
        else:
            res = nodo(self.val / other, [self], '/')

        def backward():
            if es_nodo:
                self.grad += res.grad / other.val
                other.grad += res.grad * (-self.val / other.val ** 2)
            else:
                self.grad += res.grad / other
        res.backward = backward
        return res

    def __pow__(self, other):
        """Potencia: d(a^b)/da = b·a^(b-1), d(a^b)/db = a^b·ln(a)."""
        es_nodo = not nodo.isScalar(other)
        if es_nodo:
            res = nodo(self.val ** other.val, [self, other], '**')
        else:
            res = nodo(self.val ** other, [self], '**')

        def backward():
            if es_nodo:
                self.grad += res.grad * other.val * self.val ** (other.val - 1)
                other.grad += res.grad * (self.val ** other.val) * math.log(self.val)
            else:
                self.grad += res.grad * other * self.val ** (other - 1)
        res.backward = backward
        return res

    # ------------------------------------------------------------------
    # Función de activación
    # ------------------------------------------------------------------
    def tanh(self):
        """Tangente hiperbólica: d tanh(x)/dx = 1 - tanh(x)²."""
        res = nodo(math.tanh(self.val), [self], 'tanh')

        def backward():
            self.grad += res.grad * (1 - res.val ** 2)
        res.backward = backward
        return res

    # ------------------------------------------------------------------
    # Operaciones con el escalar a la izquierda (3 + nodo, 2 * nodo, ...)
    # ------------------------------------------------------------------
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        """``escalar - nodo``."""
        res = nodo(other - self.val, [self], '-')

        def backward():
            self.grad -= res.grad
        res.backward = backward
        return res

    def __rtruediv__(self, other):
        """``escalar / nodo``."""
        res = nodo(other / self.val, [self], '/')

        def backward():
            self.grad += res.grad * -other * self.val ** -2
        res.backward = backward
        return res

    def __rpow__(self, other):
        """``escalar ** nodo``."""
        res = nodo(other ** self.val, [self], '**')

        def backward():
            self.grad += res.grad * (other ** self.val) * math.log(other)
        res.backward = backward
        return res

    # ------------------------------------------------------------------
    # Retropropagación
    # ------------------------------------------------------------------
    @staticmethod
    def backPropagation(currentNode):
        """Calcula el gradiente de ``currentNode`` respecto a todos sus ancestros.

        Construye un orden topológico del grafo (cada nodo aparece después de
        sus hijos) y lo recorre al revés aplicando la regla de la cadena.
        Los gradientes se *acumulan* en ``grad``; ponlos a 0 antes de cada pasada.
        """
        orden = []
        visitados = set()

        def construir(node):
            if node in visitados:
                return
            visitados.add(node)
            for hijo in node.hijos:
                construir(hijo)
            orden.append(node)

        construir(currentNode)
        currentNode.grad = 1.0
        for node in reversed(orden):
            if node.backward is not None:
                node.backward()
