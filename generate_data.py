import numpy as np

def generate_data(n=800, var=0.2, ratio=0.8):
    # generate data
    class_0_a = var * np.random.randn(n//4, 2)
    class_0_b = var * np.random.randn(n//4, 2) + (2, 2)
    class_1_a = var * np.random.randn(n//4, 2) + (0, 2)
    class_1_b = var * np.random.randn(n//4, 2) + (2, 0)

    X = np.concatenate([class_0_a, class_0_b, class_1_a, class_1_b], axis=0)
    Y = np.concatenate([np.zeros((n//2, 1)), np.ones((n//2, 1))])

    # shuffle the data
    rand_perm = np.random.permutation(n)
    X = X[rand_perm, :]
    Y = Y[rand_perm, :]

    X = X.T
    Y = Y.T

    # train test split
    X_train = X[:, :int(n*ratio)]
    Y_train = Y[:, :int(n*ratio)]

    X_test = X[:, int(n*ratio):]
    Y_test = Y[:, int(n*ratio):]

    return X_train, Y_train, X_test, Y_test
