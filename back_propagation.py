import numpy as np
import matplotlib.pyplot as plt
class BackPropagation:
    """h1 is the number of inputs and h2 is the number 
        of nodes in the hidden layer"""
    def __init__(self,h1,h2):
        self.h0 = h1
        self.h1 = h2
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
    
    def train(self,x_train,y_train,x_test,y_test):
        ## Training loop
        alpha = 0.001
        W1, W2, b1, b2 = self.init_params()
        n_epochs = 10000
        train_loss = []
        test_loss = []
        for i in range(n_epochs):
            ## forward pass
            A2, Z2, A1, Z1 = self.forward_pass(x_train, W1,W2, b1, b2)
            ## backward pass
            dW1, dW2, db1, db2 = self.backward_pass(x_train,y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)
            ## update parameters
            W1, W2, b1, b2 = self.update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha)

            ## save the train loss
            train_loss.append(self.loss(A2, y_train))
            ## compute test loss
            A2, Z2, A1, Z1 = self.forward_pass(x_test, W1, W2, b1, b2)
            test_loss.append(self.loss(A2, y_test))

            ## plot boundary
            if i %1000 == 0:
                self.plot_decision_boundary(W1, W2, b1, b2)

        ## plot train et test losses
        plt.plot(train_loss)
        plt.plot(test_loss)

        y_pred = self.predict(x_train, W1, W2, b1, b2)
        train_accuracy = self.accuracy(y_pred, y_train)
        print ("train accuracy :", train_accuracy)

        y_pred = self.predict(x_test, W1, W2, b1, b2)
        test_accuracy = self.accuracy(y_pred, y_test)
        print ("test accuracy :", test_accuracy)
