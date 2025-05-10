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
    def backward(self, next_grad): ...
    def update(self, optimizer): ...
    def accumulate(self, curr_grad): ...
    def zero_grad(self): ...


class ActivationFunction:
    def forward(self, x): ...
    def backward(self, next_grad): ...


class Loss:
    def calc_loss(self, y, pred): ...
    def backward(self, y, pred): ...


class Optimizer:
    def step(self, weight, grad): ...


class Linear(Layer):
    def __init__(self, input_size, output_size, weights=None):
        random_gen = RandomGenerator(42, 99131, 17372, 2**32)
        self.input_size = input_size + 1
        self.output_size = output_size
        self.weights = weights
        self.x = []
        self.grad = []

        if self.weights is None:
            self.weights = []
            for i in range(self.input_size):
                row = []
                for j in range(self.output_size):
                    row.append(random_gen.next())
                self.weights.append(row)

    def forward(self, x):
        out = []
        x_with_bias = x + [1]
        self.x = x_with_bias

        for i in range(self.output_size):
            val = 0
            for j in range(self.input_size):
                val += self.weights[j][i] * self.x[j]
            out.append(val)

        return out

    def backward(self, next_grad):
        curr_grad = []
        input_grad = [0.0] * self.input_size

        for i in range(self.input_size):
            row = []
            for j in range(self.output_size):
                row.append(self.x[i] * next_grad[j])
            curr_grad.append(row)

        for j in range(self.output_size):
            for i in range(self.input_size):
                input_grad[i] += self.weights[i][j] * next_grad[j]

        return curr_grad, input_grad

    def update(self, optimizer):
        optimizer.step(self.weights, self.grad)

    def accumulate(self, curr_grad):
        for i in range(self.input_size):
            for j in range(self.output_size):
                self.grad[i][j] += curr_grad[i][j]

    def zero_grad(self):
        self.grad = [
            [0 for _ in range(self.output_size)] for _ in range(self.input_size)
        ]


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.sx = []

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def forward(self, x):
        out = []
        for i in x:
            out.append(self.sigmoid(i))

        self.sx = out
        return out

    def backward(self, next_grad):
        return None, [
            next_grad[i] * self.sx[i] * (1 - self.sx[i]) for i in range(len(self.sx))
        ]


class ReLU(ActivationFunction):
    def __init__(self):
        self.x = []

    @staticmethod
    def relu(x):
        return max(0, x)

    def forward(self, x):
        self.x = x
        out = []
        for i in x:
            out.append(self.relu(i))

        return out

    def backward(self, next_grad):
        return None, [next_grad[i] if self.x[i] > 0 else 0 for i in range(len(self.x))]


class GradientDescent(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def step(self, weight, grad):
        for i in range(len(weight)):
            for j in range(len(weight[i])):
                weight[i][j] -= grad[i][j] * self.lr


class BCELoss(Loss):
    def calc_loss(self, y, pred):
        return sum(
            -(yi * math.log(predi + 1e-3) + (1 - yi) * math.log(1 - predi + 1e-3))
            for yi, predi in zip(y, pred)
        )

    def backward(self, y, pred):
        return [
            (-yi / (predi + 1e-5) + (1 - yi) / (1 - predi + 1e-5))
            for yi, predi in zip(y, pred)
        ]


class MSELoss(Loss):
    def calc_loss(self, y, pred):
        return sum((yi - predi) ** 2 for yi, predi in zip(y, pred)) / (len(y) * 2)

    def backward(self, y, pred):
        return [(predi - yi) for yi, predi in zip(y, pred)]


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, next_grad):
        for layer in self.layers[::-1]:
            curr_grad, input_grad = layer.backward(next_grad)
            if isinstance(layer, Layer):
                layer.accumulate(curr_grad)
            next_grad = input_grad

    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.zero_grad()

    def update(self, optimizer):
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.update(optimizer)


class TrainingLoop:
    def __init__(self, model, num_iter, optimizer, loss):
        self.model = model
        self.num_iter = num_iter
        self.optimizer = optimizer
        self.loss = loss

    def train(self, X, target):
        for _ in range(self.num_iter):
            self.model.zero_grad()
            for x, y in zip(X, target):
                pred = self.model.forward(x)
                loss = self.loss.calc_loss(y, pred)
                print(loss)
                loss_grad = self.loss.backward(y, pred)
                self.model.backward(loss_grad)

            self.model.update(self.optimizer)


if __name__ == "__main__":
    weight1 = [[-0.2, -0.01], [0.05, 0.06], [0.0000001, 0.05]]
    weight2 = [[-0.02], [-0.02], [0.01]]

    layers = [Linear(2, 2, weight1), Sigmoid(), Linear(2, 1, weight2), Sigmoid()]

    nn = NeuralNetwork(layers)
    lr = 5
    X = [[0, 0], [1, 0], [0, 1], [1, 1]]
    y = [[0], [1], [1], [0]]

    optim = GradientDescent(lr)
    loss = MSELoss()

    train_loop = TrainingLoop(nn, 10000, optim, loss)
    train_loop.train(X, y)
    print(nn.forward([0, 0]))
    print(nn.forward([0, 1]))
    print(nn.forward([1, 0]))
    print(nn.forward([1, 1]))
