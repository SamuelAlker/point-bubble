from tensorflow.keras import layers, Model, losses, metrics, regularizers, activations, backend
from keras.engine import data_adapter
import tensorflow as tf


def identity_block(x, filter, activation):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv1D(filter, 3, activation=activation, padding='same')(x)
    x = layers.BatchNormalization(axis=2)(x)
    x = layers.Activation(activation=activation)(x)
    # Layer 2
    x = layers.Conv1D(filter, 3, activation=activation, padding='same')(x)
    x = layers.BatchNormalization(axis=2)(x)
    # Add Residue
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def convolutional_block(x, filter, activation):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv1D(filter, 3, activation=activation, padding='same', strides=2)(x)
    x = layers.BatchNormalization(axis=2)(x)
    x = layers.Activation(activation=activation)(x)
    # Layer 2
    x = layers.Conv1D(filter, 3, activation=activation, padding='same')(x)
    x = layers.BatchNormalization(axis=2)(x)
    # Processing Residue with conv(1,1)
    x_skip = layers.Conv1D(filter, 1, activation=activation, strides=2)(x_skip)
    # Add Residue
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)

    return x


def resnet(activation, optimizer, input_shape, frames):
    # Step 1 (Setup Input Layer)
    x_input = layers.Input(shape=(frames, input_shape, 4), name="input")
    x = layers.ZeroPadding2D((0, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = layers.Conv2D(64, (frames, 7), strides=(1, 2), activation='tanh', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation=activation)(x)
    x = layers.Reshape((int((input_shape+6)/2), 64))(x)
    x = layers.Conv1D(64, 1, activation=activation)(x)
    x = layers.MaxPool1D(3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size, activation=activation)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size, activation=activation)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size, activation=activation)
    # Step 4 End Dense Network
    x = layers.AveragePooling1D(2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation=activation, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dense(input_shape*2, activation=activations.linear, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Reshape((input_shape, 2))(x)
    model = Model(x_input, x)
    # print(model.summary())

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    sum = "\n".join(stringlist)

    # model = Custom_Model.CustomModel(model)
    # model.compile(optimizer=optimizer)

    metric = metrics.MeanAbsoluteError(name="MAE")
    model.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=[metric], )
    return [model, sum]


def residual_cell(x, activation, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        x = layers.Dense(size, activation=activation)(x)
    input_size = x_skip.get_shape()[1]
    x = layers.Dense(input_size, activation=activation)(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def message_model(frames, input_number, output_number, activation, residual_cells=None):
    if residual_cells is None:
        residual_cells = [[2, 5], [2, 5]]
    x_input = layers.Input(shape=(frames, input_number), name="message_input")
    x = layers.Flatten()(x_input)
    for cell_struct in residual_cells:
        x = residual_cell(x, activation, layer_size=cell_struct[0], size=cell_struct[1])
    x = layers.Dense(output_number, activation=activation)(x)
    model = Model(x_input, x)
    return model


def interpretation_model(input_number, output_number, activation, residual_cells=None):
    if residual_cells is None:
        residual_cells = [[2, 10], [2, 10]]
    x_input = layers.Input(shape=input_number, name="interpretation_input")
    x = layers.Dense(30, activation=activation)(x_input)
    for cell_struct in residual_cells:
        x = residual_cell(x, activation, layer_size=cell_struct[0], size=cell_struct[1])
    x = layers.Dense(output_number, activation=activation)(x)
    model = Model(x_input, x)
    return model


def graph_network(frames, activation, m_input, m_output, i_output):
    m_model = message_model(frames, m_input, m_output, activation)
    m_model.compile()
    i_model = interpretation_model(m_output, i_output, activation)
    i_model.compile()
    model = CustomModel(m_model, i_model, m_output)
    model.compile()
    return model


class CustomModel(Model):
    mse = metrics.MeanSquaredError(name="MSE")

    def loss_function(self, y_true, y_pred):
        loss = losses.mean_squared_error(y_true, y_pred)
        return loss

    def __init__(self, m_model, i_model, m_model_output):
        super(CustomModel, self).__init__()
        self.m_model = m_model
        self.m_model_output = m_model_output
        self.i_model = i_model

    def call(self, inputs, training=None, mask=None):
        shape = inputs.shape
        message_sum = self.m_model(inputs[:, 1])
        for i in range(1, shape[1]):
            print("call", i)
            message_sum += self.m_model(inputs[:, i])
        message_sum += inputs[:, 0, 0]
        output = self.i_model(message_sum)
        output = tf.expand_dims(output, axis=1)
        for index in range(1, shape[1]):
            print(index)
            message_sum = self.m_model(inputs[:, 0])
            message_sum += inputs[:, index, 0]
            for i in range(1, shape[1]):
                if i != index:
                    message_sum += self.m_model(inputs[:, i])
            output = tf.concat((output, tf.expand_dims(self.i_model(message_sum), axis=1)), axis=1)
        return output

    @tf.function
    def train_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        shape = x.shape
        loss_tot = 0.

        # def grad_loss(training_data, labels, ind):
        #     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        #         tape.watch(self.i_model.trainable_variables)
        #         # tape_1.watch(self.m_model.trainable_variables)
        #         message = self.m_model(training_data[:, ind - 1], training=True)
        #         message = tf.expand_dims(message, 0)
        #         for i in range(0, shape[1] - 1):
        #             if i != ind and i != ind - 1:
        #                 message = tf.concat(
        #                     [message, tf.expand_dims(self.m_model(training_data[:, i], training=True), 0)], 0)
        #         message = tf.concat([message, tf.expand_dims(tf.cast(x[:, ind, 0], tf.float32), 0)], 0)
        #         message = backend.sum(message, axis=0)
        #         pred = self.i_model(message, training=True)
        #         loss_ = losses.mean_squared_error(labels[:, j], pred)

        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_1:
            tape_1.watch(self.i_model.trainable_variables)
            tape_1.watch(self.m_model.trainable_variables)
            index = 0
            for single in x:
                messages = self.m_model(single, training=True)
                # message_sum = tf.expand_dims(message_sum, 0)
                # for i in range(0, shape[1]-1):
                #     if i != index and i != index-1:
                #         message_sum = tf.concat([message_sum, tf.expand_dims(self.m_model(x[:, i], training=True), 0)], 0)
                col_index = 0
                for col in single:
                    message_sum = tf.concat([messages,  tf.expand_dims(tf.cast(col[0], tf.float32), 0)], 0)
                    message_sum = tf.expand_dims(backend.sum(message_sum, axis=0), 0)
                    y_pred = self.i_model(message_sum, training=True)[0]
                    y_true = tf.cast(y[index, col_index], tf.float32)
                    loss_tot += losses.mean_squared_error(y_true, y_pred)
                    col_index += 1
                index += 1
        self.optimizer.minimize(loss_tot, [self.i_model.trainable_variables, self.m_model.trainable_variables], tape=tape_1)

        # self.optimizer.apply_gradients(zip(i_model_grads, i_model_trainable))
        # self.optimizer.minimize(loss_1, self.i_model.trainable_variables, tape=tape_1)
        # with tf.GradientTape(persistent=True) as tape_2:
        #     shape = x.shape
        #     message_sum = self.m_model(x[:, 0], training=True)
        #     for i in range(1, shape[1]):
        #         message_sum += self.m_model(x[:, i], training=True)
        #     y_pred = self.i_model(message_sum, training=True)
        #     loss_2 = losses.mean_squared_error(y, y_pred)
        # self.optimizer.minimize(loss_1, self.i_model.trainable_variables, tape=tape)
        # self.optimizer.minimize(loss_2, self.m_model.trainable_variables, tape=tape_2)
        # y_pred = self(x, training=False)
        # self.mse.update_state(y, y_pred)
        # mse_result = self.mse.result()
        return {"MSE": loss_tot}

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # y_pred = self(x, training=False)
        # self.mse.update_state(y, y_pred)
        # mse_result = self.mse.result()
        return {"MSE": 1}