import math

class RandomGenerator:
    def __init__(self, seed, a, c, m):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

class Layer:
    def forward(self, x): ...
    def backward(self, next_grad, lr): ...

class Linear(Layer):
    def __init__(self, input_size, output_size, weights = None):
        random_gen = RandomGenerator(42, 99131, 17372, 2**32)
        self.input_size = input_size + 1
        self.output_size = output_size
        self.weights = weights
        self.x = []

        if self.weights is None:
            self.weights = []
            for i in range(self.input_size):
                row = []
                for j in range(self.output_size):
                    row.append(random_gen.next())
                self.weights.append(row)

    def forward(self, x):
        out = []
        x.append(1)
        self.x = x.copy()

        for i in range(self.output_size):
            val = 0
            for j in range(self.input_size):
                val += self.weights[j][i] * x[j]
            out.append(val)

        return out
    

class Sigmoid(Layer):
    def __init__(self):
        self.sx = []

    @staticmethod
    def sigmoid(x):
        return 1/(1 + math.exp(-x))

    def forward(self, x):
        out = []
        for i in x:
            out.append(self.sigmoid(i))

        self.sx = out.copy()
        return out
    
class ReLU(Layer):
    def __init__(self):
        self.x = []

    @staticmethod
    def relu(x):
        return max(0, x)

    def forward(self, x):
        self.x = x.copy()
        out = []
        for i in x:
            out.append(self.relu(i))

        self.sx = out.copy()
        return out
    

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train(self, x): ...

if __name__ == "__main__":
    weight1 = [
        [-1.189, -0.76],
        [1.189, 0.76],
        [0, 0.76]
        ] 

    weight2 = [
        [1.68], [-1.31], [0.99]
    ]

    layers = [
        Linear(2,2, weight1),
        ReLU(),
        Linear(2,1, weight2),
        ReLU()
    ]

    nn = NeuralNetwork(layers)

    print(nn.predict([0, 0]))
    print(nn.predict([0, 1]))
    print(nn.predict([1, 0]))
    print(nn.predict([1, 1]))
