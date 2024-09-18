import numpy as np


def mse(y_true,y_pred):
    return np.mean(np.power(y_true-y_pred,2))

def mse_prime(y_true,y_pred):
    return 2*(y_pred-y_true)/np.size(y_pred)



class Base:

    def __init__(self) :
        self.input  = None
        self.ouput = None

    def forward(self,input):
        pass


    def backward(self,output_gradient,learning_rate):
        pass



class Dense(Base):
    def __init__(self,input_size,output_size):
        self.weight = np.random.randn(output_size,input_size) 
        self.bias = np.random.randn(output_size,1)


    def forward(self,input):
        self.input = input
        return np.dot(self.weight,self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(output_gradient,self.input.T)
        self.weight -= learning_rate*weight_gradient
        self.bias -= learning_rate*output_gradient
        return np.dot(self.weight.T,output_gradient)


class Activation(Base):
    def __init__(self,activation,activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self,input):
        self.input = input
        return self.activation(self.input)    
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,self.activation_prime(self.input))      




class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x : 1-(tanh(x)**2)
        super().__init__(tanh,tanh_prime)

        


