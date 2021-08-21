import numpy as np
from numpy.ma.core import dot

class neuralnet:
    def __init__(self, layers, alpha=0.1):
        self.W = [] #weight
        # layers = [input, hidden,...,output]
        self.layers = layers  
        self.alpha = alpha   #learning rate

        #consider bias and output layer
        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i]+1, layers[i+1]+1) 
            self.W.append(w)
        w = np.random.randn(layers[-2]+1, layers[-1])
        self.W.append(w)

    def __repr__(self):
        return "NueralNetwork:{}".format("-".join (str(l) for l in self.layers))

    def active_fun(self, name, x):
        if name == 'sigmoid':
            return 1./(1 + np.exp(-x))

        if name == 'relu':
            return np.maximum(0,x)
        
        if name == 'tanh':
            return (np.exp(x) - np.exp(-x))/(np.exp(x) +np.exp(-x))

    def deriv(self, name, x):
        if name == 'sigmoid':
            return x*(1-x)

        if name == 'relu':
            return 0 if x <=  0 else 1
        
        if name == 'tanh':
            tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) +np.exp(-x))
            return 1 - tanh^2

    def partial(self, x, y, active_func = 'sigmoid'):
        A = [np.atleast_2d(x)]

        #foward
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.active_fun(active_func, net)
            A.append(out)

        #backpropagation
        error = A[-1] - y

        D = [error * self.deriv(active_func,A[-1])]

        for layer in np.arange(len(A)-2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.deriv(active_func,A[layer])
            D.append(delta)

        D = D[::-1]

        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def caculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X)
        loss = 0.5*(sum(predictions-targets)**2)
        return loss


    def fit(self, X, y, epochs = 10, snapshot = 10, active_func = 'sigmoid'):
        #bias
        X = np.c_[X,np.ones(X.shape[0])] 

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.partial(x, target, active_func)
            if epoch == 0 or epoch % snapshot == 0:
                loss = self.caculate_loss(X ,y)
                print("epoch = {}, loss = {}".format(epoch+1, loss))

    def predict(self, X, active_func = 'sigmoid'):
        res = np.atleast_2d(X)

        for layer in np.arange(0, len(self.W)):
            p = np.dot(res,self.W[layer])
            res = self.active_fun(active_func, p)
        
        return res

if __name__ == '__main__':
    nn = neuralnet([2, 2, 1])
    x = np.array([[2,3],[4,5],[6,7],[8,9]])
    y = np.array([4,8,12,16])
    nn.fit(x,y)
    x_test = np.array([10,11])
    print(nn.__repr__)
    print(nn.predict(x_test))

 
