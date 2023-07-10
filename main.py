import matplotlib.pylab as plt
import numpy as np
import timeit
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
def matrix_feed_forward_calc(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = f(z)
    return h
if __name__ == '__main__':
    '''
    x = np.arange(-8,8,0.1)
    print(x)
    f = 1/(1+np.exp(-x))
    plt.plot(x,f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    w1 = 0.5
    w2 = 1.0
    w3 = 2.0
    l1 = 'w=0.5'
    l2 = 'w=1.0'
    l3 = 'w=2.0'
    for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
        f = 1/(1+np.exp(-x*w))
        plt.plot(x, f, label =l)
    plt.xlabel('x')
    plt.ylabel('h_w(x)')
    plt.legend(loc=2)
    plt.show()
    w = 5.0
    b1 = -8.0
    b2 = 0.0
    b3 = 8.0
    l1 = 'b = -8.0'
    l2 = 'b = 0.0'
    l3 = 'b = 8.0'
    for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
        f = 1 / (1 + np.exp(-(x * w + b)))
        plt.plot(x, f, label=l)
    plt.xlabel('x')
    plt.ylabel('h_wb(x)')
    plt.legend(loc=2)
    plt.show()
    '''
    # to implement feed forward neural network
    w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
    w2 = np.zeros((1, 3))
    w2 = np.array([0.5, 0.5, 0.5])
    w2 = np.reshape(w2, (1, 3))
    b1 = np.array([0.8, 0.8, 0.8])
    b2 = np.array([0.2])
    w = [w1, w2]
    b = [b1, b2]
    x = [1.5, 2.0, 3.0]
    #print(w)
    h =simple_looped_nn_calc(3, x, w, b)
    print(h)
    h = matrix_feed_forward_calc(3, x, w, b)
    print(h)
    exec_time = timeit.timeit(lambda :simple_looped_nn_calc(3, x, w, b), number=10000)
    print(exec_time)



