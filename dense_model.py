import keras.engine.data_adapter
import numpy as np
from tensorflow.keras import layers, Model, losses, metrics, regularizers, activations, backend
from keras.engine import data_adapter
import tensorflow as tf

def dense_network_old(input_number, frames, points, activation, optimiser, input_nodes, nodes, layer_num, cell_count):
    x_input = layers.Input(shape=(input_number, frames, points), name="message_input")
    x = layers.Flatten()(x_input)
    # x = layers.Dense(nodes, activation=activation)(x)

    residual_cells = [layer_num, nodes]
    x = layers.Dense(input_nodes, activation=activation)(x)
    for i in range(cell_count):
        x = residual_cell(x, activation, layer_size=residual_cells[0], size=residual_cells[1])
        # x = layers.Dropout(0.01)(x)

    x = layers.Dense(200, activation=activations.linear)(x)
    x = layers.Reshape((100, 2))(x)
    model = Model(x_input, x)
    model.compile(optimizer=optimiser)
    # print(model.summary())
    model = CustomModel(model)
    model.compile(optimizer=optimiser)
    return model


def residual_cell(x, activation, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        x = layers.Dense(size, activation=activation)(x)
    input_size = x_skip.get_shape()[1]
    x = layers.Dense(input_size, activation=activation)(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def interpretation_model(input_number, output_number, activation, residual_cells=None):
    if residual_cells is None:
        residual_cells = [[2, 20], [2, 20], [2, 20], [2, 20], [2, 20], [2, 20]]
    x_input = layers.Input(shape=input_number, name="interpretation_input")
    x = layers.Dense(30, activation=activation)(x_input)
    for cell_struct in residual_cells:
        x = residual_cell(x, activation, layer_size=cell_struct[0], size=cell_struct[1])
        x = layers.Dropout(0.04)(x)
    x = layers.Dense(output_number, activation=activations.linear)(x)
    model = Model(x_input, x)
    return model


class CustomModel(Model):
    def __init__(self, i_model):
        super(CustomModel, self).__init__()
        self.i_model = i_model

    loss_tracker = metrics.Mean(name="loss")
    # MAE_tracker = metrics.MeanAbsoluteError(name="MAE")
    # @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.i_model(inputs, training)

    def call_2(self, inputs, training=None, mask=None):
        return self.i_model(inputs, training)

    def train_step(self, data):
        input_data, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_1:
            tape_1.watch(self.i_model.trainable_variables)
            y_pred = self(input_data, training=True)
            loss_tot = losses.mean_squared_error(y, y_pred)
        self.optimizer.minimize(loss_tot, [self.i_model.trainable_variables], tape=tape_1)
        self.loss_tracker.update_state(loss_tot)
        # self.MAE_tracker.update_state(y, y_pred)
        loss = self.loss_tracker.result()
        # MAE = self.MAE_tracker.result()
        return {"loss": loss, "loss_tot": loss_tot}

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # y_pred = self(x, training=False)
        # self.mse.update_state(y, y_pred)
        # mse_result = self.mse.result()
        return {"MSE": 1}