import numpy as np
def f(x):
    return 1 / (1 + np.exp(-x))
def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h
if __name__ =='__main__':
    w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
    w2 = np.zeros((1, 3))
    w2 = np.array([0.5, 0.5, 0.5])
    w2 = np.reshape(w2, (1, 3))
    b1 = np.array([0.8, 0.8, 0.8])
    b2 = np.array([0.2])
    w = [w1, w2]
    b = [b1, b2]
    x = [1.5, 2.0, 3.0]
    h = matrix_feed_forward_calc(3, x, w, b)
    print(h)