import numpy as np
from neurogenesis.models import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy
from neurogenesis.models import Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop

from nnfs.datasets import spiral_data
import nnfs

nnfs.init()

X,y = spiral_data(100, 3)

        

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)

loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)
# optimizer = Optimizer_Adagrad(decay=1e-4)
optimizer = Optimizer_RMSprop(learning_rate=0.02, decay=1e-4, rho=0.99)

for epoch in range(10001):

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)        

    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch} '+
              f' accuracy: {accuracy:.3f} '+
              f' loss: {loss:.3f} ' +
              f' lr: {optimizer.current_learning_rate:.3f} ' )

    #Backward pass

    loss_activation.backward(loss_activation.output, y)
    dense2.backwards(loss_activation.dinputs)
    activation1.backwards(dense2.dinputs)
    dense1.backwards(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()