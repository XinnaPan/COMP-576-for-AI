__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A 3-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################
class NeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        
        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W1 = np.random.randn(self.nn_input_dim, self.nn_hidden_dim) / np.sqrt(self.nn_input_dim)
        self.b1 = np.zeros((1, self.nn_hidden_dim))
        self.W2 = np.random.randn(self.nn_hidden_dim, self.nn_output_dim) / np.sqrt(self.nn_hidden_dim)
        self.b2 = np.zeros((1, self.nn_output_dim))

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''
        
        # YOU IMPLMENT YOUR actFun HERE

        if type == 'tanh' :
            a = (np.exp(z)-np.exp(-z)) / (np.exp(z)+np.exp(-z))
        elif type == 'sigmoid' :
            a = 1.0/(1.0+np.exp(-z))
        elif type == 'relu':
            a = max(0.0,z)
        return a

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        a=z.copy()
        if type == 'tanh' :
            return 1-pow(a,2)
        elif type == 'sigmoid':
            return a(1-a)
        elif type == 'relu':
            if z < 0 :
                a=0
            elif z >= 0:
                a=1


        return a

    def softmax(self,y):
        
        '''ee=np.exp(y)
        sum=np.sum(ee)
        return ee/sum
        '''
        c=y.copy()
        c=c.T
        c-=np.max(c,axis=0)
        x=np.exp(c)/np.sum(np.exp(c),axis=0)
        return x.T
        '''
        y -= np.max(y)
        return np.exp(y)/np.sum(np.exp(y))
        '''
        

    def feedforward(self, X, actFun):
        '''
        feedforward builds a 3-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE

        self.z1 = np.dot(X,self.W1)+self.b1
        self.a1 = actFun(self.z1)
        self.z2 = np.dot(self.a1,self.W2)+self.b2      
        self.probs = self.softmax(self.z2)

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss = -np.sum(np.log(self.probs[np.arange(num_examples),y]+1e-7))

        # Add regulatization term to loss (optional)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW1, dL/b1, dL/dW2, dL/db2
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        num_examples = len(X)
        # d = (self.probs - y)/num_examples
        if(y.size == self.probs.size) :
            d = (self.probs - y)/num_examples
        else:
            d=self.probs.copy()
            d[np.arange(num_examples),y] -=1
            d /= num_examples

        dW2 = np.dot(self.a1.T,d)
        db2 = np.sum(d,axis=0)

        
        d = np.multiply(np.dot(d,self.W2.T),self.diff_actFun(self.a1,self.actFun_type))
        dW1 = np.dot(X.T,d)
        db1 = np.sum(d,axis=0)
        return dW1, dW2, db1, db2

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW1, dW2, db1, db2 = self.backprop(X, y)

            # Add derivatives of regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            # Gradient descent parameter update
            self.W1 += -epsilon * dW1
            self.b1 += -epsilon * db1
            self.W2 += -epsilon * dW2
            self.b2 += -epsilon * db2

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)

class Layer:
    def __init__(self, w, b,act=True):
        self.x = None
        self.weight = w
        self.bias = b
        self.x_de = None
        self.weight_de = None
        self.bias_de = None
        self.act=act
        self.Y=None

    def feedforward(self, x):
        self.x = x
        self.Y = np.dot(x, self.weight) + self.bias
        return self.Y

    def backprop(self, d):
        self.x_de = np.dot(d, self.weight.T)
        self.weight_de = np.dot(self.x.T, d)
        self.bias_de = np.sum(d, axis=0)
        return self.x_de


class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self,nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        super(DeepNeuralNetwork,self).__init__(nn_input_dim, nn_hidden_dim , nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0)

        self.seed=seed
        self.params = self.initParams()
        self.hiddenLayers = self.initHiddenLayers()
        #self.lastLayer = self.initOutputLayer()

    def feedforward(self, X, actFun):
        y = X.copy()
        for layers in self.hiddenLayers:
            y = layers.feedforward(y)
            if layers.act == True:
                y= self.actFun(self.actFun_type)
        y = self.softmax(y)
        return y

    def diff_softmax(self,Y,y):
        num_examples = y.shape[0]
        if(Y.size == y.size):
            d = (Y-y) / num_examples
        else:
            d = Y.copy()
            d[np.arange(num_examples),y] -=1
            d /= num_examples
        return d

    def backprop(self, Y, y):
        d = self.diff_softmax(Y,y)
        # Reverse order for derivation of each parameter of the hidden layer
        for index in range(len(self.hiddenLayers)-1, -1, -1):
            if self.hiddenLayers[index].act == True :
                d = d * self.diff_actFun(self.hiddenLayers[index].Y,self.actFun_type)                
            d = self.hiddenLayers[index].backprop(d)
              

    def calculate_loss(self, Y, y):
        if (Y.ndim == 1):
            y = y.reshape(1, y.size)
            Y = Y.reshape(1, Y.size)

        if (Y.size == y.size):
            y = y.argmax(axis=1)

        num_examples = Y.shape[0]
        data_loss = -np.sum(np.log(Y[np.arange(num_examples), y] + 1e-7)) 
        w=[]
        for layers in self.hiddenLayers:
            w.append(layers.weight)
        data_loss += self.reg_lambda / 2 * (np.sum(np.square(w)))
        return (1. / num_examples) * data_loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True,lr=0.1):
        Y = self.feedforward(X,self.actFun_type)
        
        self.loss=self.calculate_loss(Y,y)

        self.backprop(Y,y)
        for layer in self.hiddenLayers:
            layer.weight -= layer.weight_de * lr
            layer.bias -= layer.bias_de * lr   
        
         


        



    

                   
            

    def initHiddenLayers(self):
        layers=[]
        for index, value in enumerate(self.params):
            weight = value["weight"]
            bias = value["bias"]
            layer = None
            if (index == len(self.params) - 1):
                layer = Layer(weight, bias,False)
            else:
                layer = Layer(weight, bias,True)
            layers.append(layer)
        return layers

    def initParams(self):
        params = []
        layerSizeList = [self.nn_input_dim
                         ] + self.nn_input_dim + [self.nn_output_dim]
        for index, value in enumerate(layerSizeList):
            if (index > 0):
                prevSize = layerSizeList[index - 1]
                curSize = value
                param = self.initLayerParam(prevSize, curSize)
                params.append(param)
        return params

    def initLayerParam(self, inputSize, outputSize):
        param = {}
        param["weight"] =np.random.randn(inputSize, outputSize) / np.sqrt(inputSize)    
        param["bias"] = np.zeros((1,outputSize))
        return param























def main():
    # generate and visualize Make-Moons dataset
    X, y = generate_data()
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

    model = NeuralNetwork(nn_input_dim=2, nn_hidden_dim=3 , nn_output_dim=2, actFun_type='tanh')
    model.fit_model(X,y)

    model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()