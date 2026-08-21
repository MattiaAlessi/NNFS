#N.B. This repo is being built using the tutorial from the following link: https://www.youtube.com/watch?v=Wo5dMEP_BbI&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
#P.S. all the other files are the implementation of the code episode per episode, this file is the final implementation of the code, it is the main file that runs the neural network
#P.P.S. the tutorials are unfinished so I juts started using the book

import numpy as np  #import this to execute mathematical operation like exponentials etc.. very easily
import nnfs #this little library initialize the code exactly like in the tutorial (same seed for random generations etc...)
from nnfs.datasets import spiral_data #these are the exact same data of the tutorial


nnfs.init() #initialize the project

class Layer_Dense:
    """this class is the first layer of the neural network, it is a dense layer, meaning that each neuron is connected to every input"""
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) #this is the weight matrix of the layer, initialized with random values multiplied by 0.10 to keep them small
        self.biases = np.zeros((1, n_neurons)) #this is the bias vector of the layer, initialized with zeros
        
    def forward(self, inputs):
        #the formula is output = inputs * weights + biases, where * is the dot product
        self.inputs = inputs #this is to store the inputs for later use in the backward pass
        self.output = np.dot(inputs, self.weights) + self.biases #this is the forward pass of the layer, where we calculate the output by multiplying the inputs with the weights and adding the biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims = True)
        
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Activation_ReLU:
    """this class is the activation function of the first layer, it is a ReLU function, meaning that it outputs 0 for negative values and the input for positive values"""
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs) #this is the forward pass of the activation function, where we calculate the output by applying the ReLU function to the inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        
        self.dinputs[self.inputs <= 0] = 0
    
    
class Activation_Softmax:
    """this class is the activation function of the second layer, it is a Softmax function, meaning that it outputs a probability distribution over the classes"""
    def forward(self, inputs):
        #the formula is output = exp(inputs) / sum(exp(inputs)), where exp is the exponential function and sum is the sum of all values
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #this is the forward pass of the activation function, where we calculate the output by applying the Softmax function to the inputs
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #this is the normalization step of the Softmax function, where we divide each value by the sum of all values to get a probability distribution
        self.output = probabilities  #this is the final output of the activation function, which is a probability distribution over the classes
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)



class Loss:
    """this class is the loss function of the neural network, it is a base class that will be inherited by the specific loss functions"""
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    """this class is the loss function of the neural network, it is a Categorical Crossentropy function, meaning that it calculates the loss for multi-class classification problems"""
    def forward(self, y_pred, y_true): #this is the forward pass of the loss function, where we calculate the loss by comparing the predicted values with the true values
        samples = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #this is to avoid division by zero and log(0) which is undefined, we clip the predicted values to be between 1e-7 and 1-1e-7
        
        if len(y_true.shape) == 1: #this is to check if the true values are in the form of a 1D array, meaning that they are in the form of class labels
            correct_confidences = y_pred_clipped[range(samples), y_true] #this is to get the predicted values for the correct classes, we use the true values as indices to get the predicted values for the correct classes
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1) #this is to get the predicted values for the correct classes, we use the true values as a one-hot encoded vector to get the predicted values for the correct classes
        
        negative_log_likelihoods = -np.log(correct_confidences) #this is to calculate the negative log likelihoods, which is the loss for each sample, we take the negative log of the predicted values for the correct classes
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples



class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples




class Optimizer_SDG:
    def __init__(self, learning_rate = 1., decay=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases
        
    def post_update_params(self):
        self.iterations += 1
        
    
        
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases











if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU() 

    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


    optimizer = Optimizer_SDG(decay=1e-2)
    
    
    for epoch in range(10001):


        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)

        loss = loss_activation.forward(dense2.output, y)
        
        

        #print(loss_activation.output[:5])
        

        predictions = np.argmax(loss_activation.output, axis = 1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions==y)

        if not epoch % 100:
            print(f"Epoch --> {epoch}, accuracy: {accuracy:.3f}, loss: {loss:.3f}, learning rate: {optimizer.current_learning_rate}")


        #backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        
        #update of weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    # print(dense1.dweights)
    # print(dense1.dbiases)

    # print(dense2.dweights)
    # print(dense2.dbiases)