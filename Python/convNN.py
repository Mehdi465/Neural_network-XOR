from Layer import Base,Activation
import numpy as np
from scipy import signal

def binary_cross_entropy(y_true,y_pred):
    return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def binary_cross_entropy_prime(y_true,y_pred):
    return ((1-y_true)/(1-y_pred)-y_true/y_pred)/np.size(y_true)

class Convolutional(Base):
    def __init__(self,input_shape,kernel_size,depth):
        input_depth,input_height,input_width = input_shape 
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.ouput_shape = (depth,input_height+1,input_width-kernel_size+1)
        self.kernels_shape = (depth,input_depth,kernel_size,kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.ouput_shape)


    def forward(self,input):
        self.input = input 
        self.ouput = np.copy(self.biases)  
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.ouput_shape[i] += signal.correlate2d(self.input[j],self.kernels[i,j],"valid")

        return self.ouput 



    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth): 
                kernel_gradient[i,j] = signal.correlate2d(self.input[j],output_gradient[i,"valid"])
                input_gradient[j] = signal.convolve2d(output_gradient[i],self.kernel[i,j],"full")

        self.kernels -= learning_rate*kernel_gradient
        self.biases -= learning_rate*output_gradient

        return input_gradient
    
    
class Reshape(Base):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input,self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient,self.input_shape)
    

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1/(1+ np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x)*(1-sigmoid(x))  
        super().__init__(sigmoid,sigmoid_prime)
    
