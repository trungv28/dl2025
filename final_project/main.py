from model import Conv2D, AvgPool2D, MaxPool2D, Flatten, Linear, ReLU, Sigmoid, CELoss, GradientDescent, NeuralNetwork, TrainingLoop, random_shuffle, RandomGenerator
from tensor import Tensor
from helper import load_data, arg_max, load_config


X_train, y_train = load_data("optdigits.tra")
X_test, y_test = load_data("optdigits.tes")
num_classes = 10

y_train_oh = [[1 if i == label else 0 for i in range(num_classes)] for label in y_train]

layers = load_config('config.txt')
model = NeuralNetwork(layers)
optimizer = GradientDescent(lr=0.2)
loss_fn = CELoss()

batch_size = 64
num_epochs = 30
train_loop = TrainingLoop(model, num_epochs, optimizer, loss_fn, batch_size)
train_loop.train(X_train, y_train_oh)

correct = 0
total = 0

for x, y in zip (X_test, y_test):
    X_tensor = Tensor(x)
    X_tensor = X_tensor.reshape(1, 1, *X_tensor.shape)
    pred = arg_max(model.forward(X_tensor).reshape(num_classes).data)
    total += 1
    if y == pred:
        correct += 1
print(correct/total)