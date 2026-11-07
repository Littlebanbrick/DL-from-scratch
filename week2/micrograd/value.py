import math

class Value:

    def __init__(self, data):
        self.data = data        # The value stored
        self.grad = 0.0         # The gradient
        self.label = ''         # The name
        self.children = []      # Who made this varient up?
        self.op = ''            # How?
        # `_backward` here, not `backward`
        # That's because Value class already have a `backward` method
        self._backward = lambda: None

    def __repr__(self):
        return f"Value[label: {self.label}, data: {self.data}]"

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data)
        out.children.append(self)
        out.children.append(other)
        out.op = '+'

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data)
        out.children.append(self)
        out.children.append(other)
        out.op = '*'

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        res = math.exp(self.data)

        out = Value(res)
        out.children.append(self)
        out.op = 'exp'

        def _backward():
            self.grad += res * out.grad

        out._backward = _backward
        return out

    # tanh(x) = (e**(2x)-1)/(e**(2x)+1)
    # dtanh(x) = 1 - (tanh(x))**2
    def tanh(self):
        res = math.tanh(self.data)

        out = Value(res)
        out.children.append(self)
        out.op = 'tanh'

        def _backward():
            self.grad += (1 - res**2) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        # other should only be int or float
        assert isinstance(other, (int, float))

        res = self.data ** other
        out = Value(res)
        out.children.append(self)
        out.op = 'pow'

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other**(-1)

    def __rtruediv__(self, other):
        return other * self**(-1)


    def relu(self):
        out = Value(self.data if self.data > 0 else 0)
        out.children.append(self)
        out.op = 'ReLU'

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out


    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

