def f(w0, w1, xi, yi):
    return 1/2 * (w1 * xi + w0 - yi) ** 2

def f_w0(w0, w1, xi ,yi):
    return w1 * xi + w0 - yi

def f_w1(w0, w1, xi, yi):
    return xi * (w1 * xi + w0 - yi)

def gradient_descent(w0, w1, lr, f, f_w0, f_w1, x, y, max_step = 100, tol = 1e-5):
    N = len(x)
    f_val = sum(f(w0, w1, xi, yi) for xi, yi in zip(x, y)) / N
    print(f_val)

    for i in range(max_step): 
        w0_grad = sum(f_w0(w0, w1, xi, yi) for xi, yi in zip(x, y)) / N
        w1_grad = sum(f_w1(w0, w1, xi, yi) for xi, yi in zip(x, y)) / N

        w0 -= lr * w0_grad
        w1 -= lr * w1_grad
        new_f_val = sum(f(w0, w1, xi, yi) for xi, yi in zip(x, y)) / N
        print(new_f_val)

        if abs(f_val - new_f_val) < tol:
            return w0, w1
        
        f_val = new_f_val
    return w0, w1

if __name__ == "__main__":
    x = []
    y = []
    with open('lr.csv', 'r') as file:
        for line in file.readlines():
            xi, yi = line.split(',')
            x.append(float(xi))
            y.append(float(yi))

    w0_init = 0
    w1_init = 0
    lr = 1e-4

    w0, w1 = gradient_descent(w0_init, w1_init, lr, f, f_w0, f_w1, x, y, 10)
