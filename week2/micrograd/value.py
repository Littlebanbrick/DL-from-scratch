import math

class Value:

    def __init__(self, data):
        self.data = data        # The value stored
        self.grad = 0.0         # The gradient
        self.label = ''         # The name
        self.children = []      # Who made this varient up?
        self.op = ''            # How?
        self.backward = lambda: None

    def __repr__(self):
        return f"Value[label: {self.label}, data: {self.data}]"

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data)
        out.children = [self, other]
        out.op = '+'

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out.backward = backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data)
        out.children = [self, other]
        out.op = '*'

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out.backward = backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def exp(self):
        res = math.exp(self.data)

        out = Value(res)
        out.children = [self]
        out.op = 'exp'

        def backward():
            self.grad = res

        out.backward = backward
        return out


    def relu(self):
        out = Value(self.data if self.data > 0 else 0)
        out.children = [self]
        out.op = 'ReLU'
        return out
