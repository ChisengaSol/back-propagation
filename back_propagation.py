import numpy as np
import matplotlib.pyplot as plt

# generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape

# shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]

X = X.T
Y = Y.T
X.shape, Y.shape

# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]

X_train.shape

plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
plt.show()


class BackPropagation:
    def __init__(self):
        self.h0 = 2
        self.h1 = 10
        self.h2 = 1
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss(self,y_pred, Y):
        return -np.mean(Y * np.log(y_pred)+(1-Y)*np.log(1-y_pred))


    ## Initialize parameters
    def init_params(self):
        W1 = np.random.randn(self.h1, self.h0) 
        W2 = np.random.randn(self.h2, self.h1) 
        b1 = np.random.randn(self.h1, self.h2) 
        b2 = np.random.randn(self.h2, self.h2) 

        return W1, W2, b1, b2

    ## Forward pass
    def forward_pass(self,X, W1,W2, b1, b2):
        Z1 = W1.dot(X) + b1
        A1 = self.sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1

    ## Backward pass
    def backward_pass(self,X,Y, A2, Z2, A1, Z1, W1, W2, b1, b2):
        dW2 = -(Y-A2)@A1.T
        db2 = -np.sum(Y-A2, axis=1, keepdims=True)
        dW1 = -(W2.T*(Y-A2)*A1*(1-A1))@X.T
        db1 = -(A1*(1-A1))@(Y-A2).T*W2.T
        return dW1, dW2, db1, db2

    ## Accuracy
    def accuracy(self,y_pred, y):
        return np.mean(y == y_pred) * 100


    def predict(self,X,W1,W2, b1, b2):
        A2, _, _, _ =self.forward_pass(X, W1,W2, b1, b2)
        output = (A2 > 0.5).astype(int)
        return output

    ## Update parameters
    def update(self,W1, W2, b1, b2,dW1, dW2, db1, db2, alpha ):
        W1 = W1 - alpha * dW1
        W2 = W2 - alpha * dW2
        b1 = b1 - alpha * db1
        b2 = b2 - alpha * db2

        return W1, W2, b1, b2

    ## Plot decision boundary
    def plot_decision_boundary(self,W1, W2, b1, b2):
        x = np.linspace(-0.5, 2.5,100 )
        y = np.linspace(-0.5, 2.5,100 )
        xv , yv = np.meshgrid(x,y)
        xv.shape , yv.shape
        X_ = np.stack([xv,yv],axis = 0)
        X_ = X_.reshape(2,-1)
        A2, Z2, A1, Z1 = self.forward_pass(X_, W1, W2, b1, b2)
        plt.figure()
        plt.scatter(X_[0,:], X_[1,:], c= A2)
        plt.show()

    def train(self):
        ## Training loop
        alpha = 0.001
        W1, W2, b1, b2 = self.init_params()
        n_epochs = 10000
        train_loss = []
        test_loss = []
        for i in range(n_epochs):
            ## forward pass
            A2, Z2, A1, Z1 = self.forward_pass(X_train, W1,W2, b1, b2)
            ## backward pass
            dW1, dW2, db1, db2 = self.backward_pass(X_train,Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
            ## update parameters
            W1, W2, b1, b2 = self.update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha)

            ## save the train loss
            train_loss.append(self.loss(A2, Y_train))
            ## compute test loss
            A2, Z2, A1, Z1 = self.forward_pass(X_test, W1, W2, b1, b2)
            test_loss.append(self.loss(A2, Y_test))

            ## plot boundary
            if i %1000 == 0:
                self.plot_decision_boundary(W1, W2, b1, b2)

        ## plot train et test losses
        plt.plot(train_loss)
        plt.plot(test_loss)

        y_pred = self.predict(X_train, W1, W2, b1, b2)
        train_accuracy = self.accuracy(y_pred, Y_train)
        print ("train accuracy :", train_accuracy)

        y_pred = self.predict(X_test, W1, W2, b1, b2)
        test_accuracy = self.accuracy(y_pred, Y_test)
        print ("test accuracy :", test_accuracy)

if __name__ == '__main__':
    bp = BackPropagation()
    bp.train()

