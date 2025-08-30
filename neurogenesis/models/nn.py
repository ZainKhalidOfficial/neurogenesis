import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# nnfs.init() #initializes the random number generator and default numpy types for proper reproducibility

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons) 
        #0.01 so that no random number is too large (>1) or too small (< -1)
        #And (n_inputs, n_neurons) prevents having to transpose the weights later during dot product
        
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs #need them when calculating the partial derivative with respect to weights during backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases
        
    
    def backwards(self, dvalues):
        # Calculate gradients on Parameters    
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Calculate gradient on the values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backwards(self, dvalues):
        
        self.dinputs = dvalues.copy() # copy values to not change the original values
        self.dinputs[self.inputs <= 0] = 0  # Derivative of ReLU is 1 for positive inputs and 0 for negative inputs
        
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # - max so that large values becomes small (0 here), so our exponentials in next line does not become too large to cause overflow error
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Calculate gradient for each sample
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # In confidence range (0 to 1), changes 0 to 1e-7 and 1 to 1-1e-7 to avoid log(0) error and log(1) == 0 which is problamatic for loss calculation 
        
        if len(y_true.shape) == 1: # if y_true has scalar values not one-hot encoded
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # if y_true is one-hot encoded
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        
        #Number of samples
        n_samples = len(dvalues)
        
        #Use first sample to determine number of labels
        n_labels = len(dvalues[0])

        # If labels are sparse, convert them to one-hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
            
        # Calculate gradient
        self.dinputs= - y_true / dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / n_samples 
        
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    #Create activation and loss function objects
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
            y_true = np.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()
        
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
class Optimizer_SGD: # Stochastic Gradient Descent
    
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.): #lr of 1 is default for SGD
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations)) # 1 is added to avoid division by very small number i.e, 1 / 0.001 = 1000
            
        
    def update_params(self, layer):
        
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        #Vanilla SGD
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
            
        layer.weights += weight_updates
        layer.biases += bias_updates
        
        
    def post_update_params(self):
        self.iterations += 1
        
class Optimizer_Adagrad: # Adaptive Gradient Descent
    
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7): #epsilon is a small number to avoid division by zero
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    def pre_update_params(self):    
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations)) # 1 is added to avoid division by very small number i.e, 1 / 0.001 = 1000
            
        
    def update_params(self, layer):
                
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        #Vanilla SGD + Normalization
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
            
        
        
    def post_update_params(self):
        self.iterations += 1
        
class Optimizer_RMSprop: # Root Mean Square Propagation
    
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9): #epsilon is a small number to avoid division by zero
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    def pre_update_params(self):    
        if self.decay:
            self.current_learning_rate = self.learning_rate * ( 1. / ( 1. + self.decay * self.iterations)) # 1 is added to avoid division by very small number i.e, 1 / 0.001 = 1000
        
    def update_params(self, layer):
                
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        layer.weight_cache = self.rho * layer.weight_cache + ( 1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        #Vanilla SGD + Normalization
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
            
        
        
    def post_update_params(self):
        self.iterations += 1
               

# X,y = spiral_data(100, 3)

        

# dense1 = Layer_Dense(2, 64)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(64, 3)

# loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# # optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
# # optimizer = Optimizer_Adagrad(decay=1e-4)
# optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-4, rho=0.99)

# for epoch in range(10001):

#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)

#     loss = loss_activation.forward(dense2.output, y)

#     predictions = np.argmax(loss_activation.output, axis=1)

#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)        

#     accuracy = np.mean(predictions == y)

#     if not epoch % 100:
#         print(f'epoch: {epoch} '+
#               f' accuracy: {accuracy:.3f} '+
#               f' loss: {loss:.3f} ' +
#               f' lr: {optimizer.current_learning_rate:.3f} ' )

#     #Backward pass

#     loss_activation.backward(loss_activation.output, y)
#     dense2.backwards(loss_activation.dinputs)
#     activation1.backwards(dense2.dinputs)
#     dense1.backwards(activation1.dinputs)

#     optimizer.pre_update_params()
#     optimizer.update_params(dense1)
#     optimizer.update_params(dense2)
#     optimizer.post_update_params()


# # softmax_outputs = np.array([[0.7, 0.1, 0.2],
# #                             [0.1, 0.5, 0.4],
# #                             [0.02, 0.9, 0.08]])

# # class_targets = np.array([0, 1, 1])

# # softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
# # softmax_loss.backward(softmax_outputs, class_targets)
# # dvalue1 = softmax_loss.dinputs

# # activation = Activation_Softmax()
# # activation.output = softmax_outputs
# # loss = Loss_CategoricalCrossentropy()
# # loss.backward(softmax_outputs, class_targets)
# # activation.backward(loss.dinputs)
# # dvalue2 = activation.dinputs

# # print("Dvalue from Activation_Softmax_Loss_CategoricalCrossentropy:", dvalue1)
# # print("Dvalue from Activation_Softmax and Loss_CategoricalCrossentropy:", dvalue2)

