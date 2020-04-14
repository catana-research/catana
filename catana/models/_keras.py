# -*- coding: utf-8 -*-
"""Single line module summary.

Longer file description if required.
"""
import datetime

from keras import models
from keras.callbacks import ModelCheckpoint, TensorBoard

def neural_network():
    model = None
    return



def create_model(n_input, hidden_layers, n_output, lr=0.01):
    """Create a neural network with the specified number of input, hidden and output nodes

    - hidden_layers: Array of int, representing the number of nodes per hidden layer
    """
    model = models.Sequential()
    model.add(models.Dense(hidden_layers[0], activation='relu', input_dim=n_input))  # Add activation layers
    if len(hidden_layers) > 1:
        for n_hidden in hidden_layers[1:]:
            model.add(models.Dense(n_hidden, activation='sigmoid'))  # Add hidden layer
    model.add(models.Dense(n_output, activation='linear'))

    from keras.optimizers import RMSprop, Adam
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # For a mean squared error regression problem
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model


def create_tensor_board():
    """
    cd /d R:\Projects\MachineLearning
    tensorboard --logdir=R:\Projects\MachineLearning\Graph\
    """
    log_dir = "Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

"""
Example
-------

    n_features = np.shape(X)[1]

    x_train = X
    y_train = y

    #checkpoint = ModelCheckpoint(weight_save_path, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='max')
    tb_callback = create_tensor_board()
    callbacks_list = [tb_callback]

    model = create_model(n_features, hidden_layers, n_output)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        verbose=verbose_flag, validation_split=validation_split,
                        callbacks=callbacks_list)

    model.summary()
    model.get_weights()

"""
