import math

def sigmoid(x):
    return 1/(1 + math.exp(-x))


def loss(yi, y_hati):
    return -(yi * y_hati - math.log(1 + math.exp(y_hati)))

def f_w0(xi, yi, y_hati):
    return sigmoid(y_hati) - yi

def f_w1(xi, yi, y_hati):
    return (sigmoid(y_hati) - yi) * xi[0]

def f_w2(xi, yi, y_hati):
    return (sigmoid(y_hati) - yi) * xi[1]


def gradient_descent(w0, w1, w2, lr, x, y, max_step = 100, tol = 1e-5):
    N = len(x)
    y_hat = [(w1 * xi[0] + w2 * xi[1] + w0) for xi in x]
    f_val = sum(loss(yi, y_hati) for yi, y_hati in zip(y, y_hat)) / N
    print(f_val)

    for i in range(max_step): 
        w0_grad = sum(f_w0(xi, yi, y_hati) for xi, yi, y_hati in zip(x, y, y_hat)) / N
        w1_grad = sum(f_w1(xi, yi, y_hati) for xi, yi, y_hati in zip(x, y, y_hat)) / N
        w2_grad = sum(f_w2(xi, yi, y_hati) for xi, yi, y_hati in zip(x, y, y_hat)) / N

        w0 -= lr * w0_grad
        w1 -= lr * w1_grad
        w2 -= lr * w2_grad

        y_hat = [(w1 * xi[0] + w2 * xi[1] + w0) for xi in x]
        new_f_val = sum(loss(yi, y_hati) for yi, y_hati in zip(y, y_hat)) / N
        print(new_f_val)

        if abs(f_val - new_f_val) < tol:
            return w0, w1, w2
        
        f_val = new_f_val
    return w0, w1, w2


if __name__ == "__main__":
    x = []
    y = []
    with open('loan2.csv', 'r') as file:
        file.readline()
        for line in file.readlines():
            xi1, xi2, yi = map(float, line.split(','))
            x.append([xi1, xi2])
            y.append(yi)

    w0_init = 0.01
    w1_init = -0.01
    w2_init = 0.01
    lr = 0.1

    w0, w1, w2 = gradient_descent(w0_init, w1_init, w2_init, lr, x, y)
