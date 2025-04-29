def gradient_descent(x, lr, f,f_prime, max_step = 50, tol = 1e-5):
    f_val = f(x)
    print(f_val)
    for i in range(max_step): 
        x -= lr * f_prime(x)
        new_f_val = f(x)
        print(new_f_val)
        if abs(f_val - new_f_val) < tol:
            return x
        f_val = new_f_val
    return x
    
def test_f(x):
    return x**2

def test_f_prime(x):
    return 2 * x

if __name__ == '__main__':
    init_x = 10
    lr = 1.1
    f = test_f
    f_prime = test_f_prime

    x = gradient_descent(init_x, lr, test_f, test_f_prime)