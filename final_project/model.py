import math
from tensor import Tensor

class RandomGenerator:
    def __init__(self, seed, a, c, m):
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return ((self.state / self.m) * 2 - 1) * 0.5


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
        self.x = None
        self.grad = None

        if self.weights is None:
            self.weights = Tensor(
                [
                    [random_gen.next() for _ in range(self.output_size)]
                    for _ in range(self.input_size)
                ]
            )
        else:
            self.weights = Tensor(weights)
        self.zero_grad()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, x.shape[0])
        x_with_bias = Tensor([xi + [1] for xi in x.data])
        self.x = x_with_bias
        out = x_with_bias.dot(self.weights)
        return out

    def backward(self, next_grad):
        x = self.x
        curr_grad = x.transpose2d().dot(next_grad)
        input_grad = next_grad.dot(self.weights.transpose2d())
        input_grad = Tensor([row[:-1] for row in input_grad.data])
        return curr_grad, input_grad

    def update(self, optimizer):
        optimizer.step(self.weights, self.grad)

    def accumulate(self, curr_grad):
        if self.grad is None:
            self.zero_grad()
        self.grad = self.grad + curr_grad

    def zero_grad(self):
        self.grad = Tensor.fill(self.input_size, self.output_size, val=0)


class Conv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        weights=None,
        use_bias = True,
        bias = None
    ):
        random_gen = RandomGenerator(42, 99131, 17372, 2**32)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.x = None
        self.grad = None

        if weights is None:
            weights_data = [
                [
                    [
                        [random_gen.next() for _ in range(self.kernel_size)]
                        for _ in range(self.kernel_size)
                    ]
                    for _ in range(self.in_channels)
                ]
                for _ in range(self.out_channels)
            ]
            self.weights = Tensor(weights_data)
        else:
            self.weights = Tensor(weights)

        self.use_bias = use_bias
        if self.use_bias:
            self.bias_grad = None
            if bias is None:
                self.bias = Tensor.fill(out_channels, val=0)
            else:
                self.bias = Tensor(bias)
        self.zero_grad()

    def forward(self, x):
        self.x = x

        if self.padding > 0:
            x = x.add_pad(self.padding)

        kernel = self.weights
        if self.dilation > 1:
            kernel = kernel.dilate(self.dilation)
        out = x.conv(kernel, stride=self.stride)
        
        if self.use_bias:
            # broadcasting sum -> to be implemented in Tensor
            batch, ch, h, w = out.shape
            for b in range(batch):
                for c in range(ch):
                    for i in range(h):
                        for j in range(w):
                            out.data[b][c][i][j] += self.bias.data[c]
        
        return out

    def backward(self, next_grad):
        x = self.x
        x = x.T()
        if self.padding > 0:
            x = x.add_pad(self.padding)
        next_grad_T = next_grad.T()
        curr_grad = x.conv(next_grad_T).T()

        if self.use_bias:
            batch, ch, h, w = next_grad.shape
            bias_grad = Tensor.fill(self.out_channels, val=0).data

            # sum over axes -> to be implemented in Tensor
            for b in range(batch):
                for c in range(ch):
                    for i in range(h):
                        for j in range(w):
                            bias_grad[c] += next_grad.data[b][c][i][j]
            curr_grad = (curr_grad, Tensor(bias_grad))

        w_flipped = self.weights.T().flip()
        if self.stride > 1:
            next_grad = next_grad.dilate(self.stride)

        next_grad = next_grad.add_pad(self.kernel_size - 1)
        input_grad = next_grad.conv(w_flipped)

        if self.padding > 0:
            input_grad = input_grad.unpad(self.padding)

        return curr_grad, input_grad

    def update(self, optimizer):
        optimizer.step(self.weights, self.grad)
        if self.use_bias:
            optimizer.step(self.bias, self.bias_grad)

    def accumulate(self, curr_grad):
        if self.grad is None:
            self.zero_grad()
        if not self.use_bias:
            self.grad = self.grad + curr_grad
        else:
            grad, bias_grad = curr_grad
            
            self.grad = self.grad + grad
            self.bias_grad = self.bias_grad + bias_grad

    def zero_grad(self):
        self.grad = Tensor.fill(*self.weights.shape, val=0)
        if self.use_bias:
            self.bias_grad = Tensor.fill(self.out_channels, val=0)


class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None
        self.mask = None

    def forward(self, x):
        self.x = x
        out, mask = x.max_pool(kernel_size=self.kernel_size, stride=self.stride)
        self.mask = mask
        return out

    def backward(self, next_grad):
        input_grad = Tensor.fill(*self.x.shape, val=0.0).data
        batch, ch, o_h, o_w = next_grad.shape

        for b in range(batch):
            for c in range(ch):
                for i in range(o_h):
                    for j in range(o_w):
                        grad_val = next_grad.data[b][c][i][j]
                        mask_window = self.mask.data[b][c][i][j]

                        idx = 0
                        for ki in range(self.kernel_size):
                            for kj in range(self.kernel_size):
                                r_in = i * self.stride + ki
                                c_in = j * self.stride + kj
                                if (
                                    0 <= r_in < self.x.shape[2]
                                    and 0 <= c_in < self.x.shape[3]
                                ):
                                    input_grad[b][c][r_in][c_in] += (
                                        grad_val * mask_window[idx]
                                    )
                                idx += 1

        return None, Tensor(input_grad)


class AvgPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.x = None

    def forward(self, x):
        self.x = x
        return x.avg_pool(kernel_size=self.kernel_size, stride=self.stride)

    def backward(self, next_grad):
        input_grad = Tensor.fill(*self.x.shape, val=0.0).data
        batch, ch, o_h, o_w = next_grad.shape

        window_size = self.kernel_size**2

        for b in range(batch):
            for c in range(ch):
                for i in range(o_h):
                    for j in range(o_w):
                        grad_val = next_grad.data[b][c][i][j] / window_size
                        for ki in range(self.kernel_size):
                            for kj in range(self.kernel_size):
                                r_in = i * self.stride + ki
                                c_in = j * self.stride + kj
                                if (
                                    0 <= r_in < self.x.shape[2]
                                    and 0 <= c_in < self.x.shape[3]
                                ):
                                    input_grad[b][c][r_in][c_in] += grad_val
        return None, Tensor(input_grad)


class Flatten(Layer):
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        batch = x.shape[0]
        num_feats = 1
        for dim in x.shape[1:]:
            num_feats *= dim

        return x.flatten().reshape(batch, num_feats)

    def backward(self, next_grad):
        return None, next_grad.reshape(*self.input_shape)


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.sx = None

    @staticmethod
    def sigmoid(val):
        return 1 / (1 + math.exp(-val))

    def forward(self, x):
        out = x.activate(self.sigmoid)
        self.sx = out
        return out

    def backward(self, next_grad):
        one_tensor = Tensor.fill(*self.sx.shape, val=1.0)
        return None, next_grad * (self.sx * (one_tensor - self.sx))


class ReLU(ActivationFunction):
    def __init__(self):
        self.x = None

    @staticmethod
    def relu(val):
        return max(0, val)

    def forward(self, x):
        self.x = x
        return x.activate(self.relu)

    def backward(self, next_grad):
        return None, next_grad * self.x.activate(lambda v: 1.0 if v > 0 else 0.0)


class GradientDescent(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def step(self, weight_tensor, grad_tensor):
        new_weight_tensor = weight_tensor - grad_tensor * self.lr
        weight_tensor.data = new_weight_tensor.data


def softmax(x):
    max_x = max(x)
    exps = [math.exp(val - max_x) for val in x]
    sum_exps = sum(exps)
    return [exp / sum_exps for exp in exps]


class CELoss(Loss):
    def calc_loss(self, y_true_tensor, pred_tensor):
        probs_tensor = Tensor([softmax(row) for row in pred_tensor.data])
        log_probs = probs_tensor.activate(lambda p: -math.log(p+1e-11))

        loss_elements = y_true_tensor * log_probs

        batch_loss_sum = sum(loss_elements.flatten().data)
        return batch_loss_sum / y_true_tensor.shape[0]

    def backward(self, y_true_tensor, pred_tensor):
        probs_tensor = Tensor([softmax(row) for row in pred_tensor.data])

        grad_tensor = probs_tensor - y_true_tensor

        batch_size = y_true_tensor.shape[0]
        grad_tensor = grad_tensor * (1.0 / batch_size)

        return grad_tensor


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, next_grad):
        for layer in reversed(self.layers):
            curr_grad, input_grad = layer.backward(next_grad)

            if curr_grad is not None and hasattr(layer, "accumulate"):
                layer.accumulate(curr_grad)
            
            next_grad = input_grad

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def update(self, optimizer):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(optimizer)


def random_shuffle(arr, random_gen):
    for i in reversed(range(1, len(arr))):
        rand_val = (random_gen.next() + 0.5) 
        j = int(rand_val * (i + 1))
        arr[i], arr[j] = arr[j], arr[i]


class TrainingLoop:
    def __init__(self, model, num_iter, optimizer, loss, batch_size=8):
        self.model = model
        self.num_iter = num_iter
        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size

    def batch_iter(self, X, y):
        random_gen = RandomGenerator(42, 99131, 17372, 2**32)
        indices = list(range(len(X)))
        random_shuffle(indices, random_gen)
        for i in range(0, len(X), self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            Xb = [X[j] for j in batch_idx]
            yb = [y[j] for j in batch_idx]
            yield Xb, yb

    def train(self, X, target):
        from tqdm import tqdm
        for _ in range(self.num_iter):
            losses = []
            for Xb, yb in tqdm(self.batch_iter(X, target)):
                self.model.zero_grad()
                Xb_tensor = (
                    Tensor([[x] for x in Xb])
                    if isinstance(Xb[0][0][0], float)
                    else Tensor([[[img] for img in Xb]])
                )

                yb_tensor = Tensor(yb)
                pred = self.model.forward(Xb_tensor)
                loss = self.loss.calc_loss(yb_tensor, pred)
                losses.append(loss)
                loss_grad = self.loss.backward(yb_tensor, pred)
                self.model.backward(loss_grad)
                self.model.update(self.optimizer)

            print(sum(losses) / len(losses))
