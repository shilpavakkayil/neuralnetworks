import numpy as np
def simple_looped_nn_calc(n_layers, x, w, b):
    for l in range(n_layers - 1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        #print(w[l].shape)
        h = np.zeros((w[l].shape[0],))
        for i in range(w[l].shape[0]):
            f_sum = 0
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            f_sum += b[l][i]
            h[i] = f(f_sum)
            #print(h[i])
        #print(h)
    return h
def f(x):
    return 1 / (1 + np.exp(-x))
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
    h = simple_looped_nn_calc(3, x, w, b)
    print(h)