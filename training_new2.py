import ast
import os
import gc
import pandas as pd
import matplotlib.cm as cm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import subprocess
import numpy as np
import glob
from tqdm import tqdm
from tensorflow.keras import layers, initializers, activations, losses, metrics, optimizers, Model, callbacks, backend
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from keras.engine import data_adapter
from datetime import datetime
import pickle


def residual_cell(x, activation, initialiser_w, initialiser_b, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        # x = layers.Dense(size, activation=activation)(x)
        x = layers.Dense(size, activation=activation, kernel_initializer=initialiser_w, bias_initializer=initialiser_b)(
            x)
    input_size = x_skip.get_shape()
    # x = layers.Dense(input_size, activation=activation)(x)
    # x = ScaleLayer()(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def dense_network(input_number, points, activation, optimiser, input_nodes, nodes, layer_num, cell_count,
                  initialiser_w, initialiser_b, loss_func):
    tf.compat.v1.keras.backend.clear_session()
    x_input = layers.Input(shape=(input_number, points), name="message_input")
    x = layers.Flatten()(x_input)
    # x = layers.Dense(nodes, activation=activation)(x)

    residual_cells = [layer_num, nodes]
    x = layers.Dense(nodes, activation=activations.linear, kernel_initializer=initialiser_w,
                     bias_initializer=initialiser_b)(x)
    # x = layers.Dense(nodes, activation=activations.linear)(x)
    for i in range(cell_count):
        # x = layers.Dropout(0.05)(x)
        x = residual_cell(x, activation, initialiser_w, initialiser_b, layer_size=residual_cells[0],
                          size=residual_cells[1])

    # x = layers.Dense(200, activation=activations.linear)(x)
    x = layers.Dense(200, activation=activations.linear, kernel_initializer=initialiser_w,
                     bias_initializer=initialiser_b)(x)
    x = layers.Reshape((100, 2))(x)
    model = Model(x_input, x)
    # model.compile(optimizer=optimiser)
    # print(model.summary())
    # model = CustomModel(model)
    model.compile(optimizer=optimiser, loss=loss_func, run_eagerly=False)
    return model


class CustomModel(Model):
    loss_tracker = metrics.Mean(name="loss")

    def __init__(self, i_model, min_c_training, max_c_training, min_c_labels, max_c_labels):
        super(CustomModel, self).__init__()
        self.i_model = i_model
        self.min_c_training = min_c_training
        self.max_c_training = max_c_training
        self.min_c_labels = min_c_labels
        self.max_c_labels = max_c_labels

    def call(self, input_data, training=None, mask=None):
        # if training:
        return self.i_model(input_data, training)
        # else:
        #     y_pred = self.i_model(input_data, training)
        #     corr_x = self.data_unnormalisation(input_data[:, :, 0],
        #                                        self.min_c_training[:, :, 0],
        #                                        self.max_c_training[:, :, 0])
        #     corr_y = self.data_unnormalisation(input_data[:, :, 1],
        #                                        self.min_c_training[:, :, 1],
        #                                        self.max_c_training[:, :, 1])
        #     diff_x = self.data_unnormalisation(y_pred[:, :, 0],
        #                                        self.min_c_labels[:, :, 0, 0],
        #                                        self.max_c_labels[:, :, 0, 0])
        #     diff_y = self.data_unnormalisation(y_pred[:, :, 1],
        #                                        self.min_c_labels[:, :, 0, 1],
        #                                        self.max_c_labels[:, :, 0, 1])
        #     output = self.position_transform(corr_x, corr_y, diff_x, diff_y)
        #     return output

    def data_normalisation(self, data, min_array, max_array):
        u = 1
        l = -1
        data = ((data - min_array) / (max_array - min_array)) * (u - l) + l
        # try:
        #     for i in range(len(data[0])):
        #         data[:, i] = ((data[:, i] - min_array[i]) / (max_array[i] - min_array[i])) * (u - l) + l
        # except:
        #     data = ((data - min_array) / (max_array - min_array)) * (u - l) + l
        return data

    def data_unnormalisation(self, data, min_, max_):
        u = 1
        l = -1
        data = (data - l) * (max_ - min_) / (u - l) + min_
        return data

    def position_transform(self, x, y, dx, dy):
        shape = backend.shape(x)
        nx = backend.reshape([x + dx], shape=(shape[0], shape[1], 1))
        ny = backend.reshape([y + dy], shape=(shape[0], shape[1], 1))
        data = backend.concatenate([nx, ny], axis=2)
        return data

    def train_step(self, data):
        input_data, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        true_shape = backend.shape(y)
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape_1:
            y_pred = self(input_data, training=True)
            loss_1 = losses.mean_absolute_error(y[:, :, 0], y_pred)
            corr_x = self.data_unnormalisation(input_data[:, :, 0],
                                               self.min_c_training[:, :, 0],
                                               self.max_c_training[:, :, 0])
            corr_y = self.data_unnormalisation(input_data[:, :, 1],
                                               self.min_c_training[:, :, 1],
                                               self.max_c_training[:, :, 1])
            diff_x = self.data_unnormalisation(y_pred[:, :, 0],
                                               self.min_c_labels[:, :, 0, 0],
                                               self.max_c_labels[:, :, 0, 0])
            diff_y = self.data_unnormalisation(y_pred[:, :, 1],
                                               self.min_c_labels[:, :, 0, 1],
                                               self.max_c_labels[:, :, 0, 1])
            data_adjusted = self.position_transform(corr_x, corr_y, diff_x, diff_y)
            y_pred = self(self.data_normalisation(data_adjusted, self.min_c_training, self.max_c_training), training=True)
            loss_2 = losses.mean_absolute_error(y[:, :, 1], y_pred)
            loss_tot = loss_1

        self.optimizer.minimize(loss_tot, [self.trainable_variables], tape=tape_1)
        self.loss_tracker.update_state(loss_tot)
        loss = self.loss_tracker.result()
        # MAE = self.MAE_tracker.result()
        return {"loss": loss, "loss_tot": loss_tot, "loss_1": loss_1}

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        loss_tot = losses.mean_absolute_error(y[:, :, 0], y_pred)
        self.loss_tracker.reset_states()
        # y_pred = self(x, training=False)
        # self.mse.update_state(y, y_pred)
        # mse_result = self.mse.result()
        return {"MSE": loss_tot}


def step_decay(epoch):
    int_rate = 0.001
    l_rate = max(int_rate * 0.1 ** int(epoch / 100), 1e-6)
    if epoch < 10:
        l_rate = 0.001
    return l_rate


def lr_scheduler():
    return callbacks.LearningRateScheduler(step_decay)


def main():
    backend.set_floatx('float64')
    print("Running Training")
    activation = activations.swish
    optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    initialiser_w = initializers.VarianceScaling(scale=2.9)
    initialiser_b = initializers.RandomNormal(stddev=0.04)
    frames = 2
    points = 100
    step = 1

    today = datetime.today()
    dt_string = today.strftime("%d_%m_%Y %H_%M")
    lr = callbacks.LearningRateScheduler(step_decay)

    simulations_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # simulations_array = [0, 1]

    data_mae = creating_data_graph(frames, points, simulations_array, 3, -1, step)
    min_c_training, max_c_training = data_normalisation_constants(data_mae[0])
    min_c_labels, max_c_labels = data_normalisation_constants(data_mae[1])
    data_mae[0] = data_normalisation(data_mae[0], min_c_training, max_c_training)
    data_mae[1] = data_normalisation(data_mae[1], min_c_labels, max_c_labels)

    val_datas_mae = tf.data.Dataset.from_tensor_slices((data_mae[0][-1000:], data_mae[1][-1000:])) \
        .shuffle(buffer_size=20000) \
        .batch(batch_size=32,
               drop_remainder=True)
    datas_mae = tf.data.Dataset.from_tensor_slices((data_mae[0][:-1000], data_mae[1][:-1000])) \
        .shuffle(buffer_size=20000) \
        .batch(batch_size=32,
               drop_remainder=True)

    loss_mae = losses.mean_absolute_error
    model_mae = dense_network(100, 2,
                              activation, optimizer,
                              150, 150, 2, 12,
                              initialiser_w, initialiser_b,
                              loss_mae)
    m1 = tf.convert_to_tensor(min_c_training)
    model_mae = CustomModel(model_mae,
                            tf.convert_to_tensor(min_c_training),
                            tf.convert_to_tensor(max_c_training),
                            tf.convert_to_tensor(min_c_labels),
                            tf.convert_to_tensor(max_c_labels))
    model_mae.compile(optimizer, loss_mae, run_eagerly=True)

    # '''

    os.makedirs("save_weights/" + dt_string)

    history_mae = model_mae.fit(datas_mae,
                                epochs=310,
                                callbacks=[lr],
                                verbose=1,
                                shuffle=True,
                                batch_size=32,
                                validation_data=val_datas_mae)

    model_mae.i_model.save_weights("save_weights/"
                                   + dt_string
                                   + "/mae_loss{}.h5".format(history_mae.history["loss"][-1]))
    loss = history_mae.history['loss_tot']
    # val_loss = history_mae.history['val_loss']
    # plt.figure(figsize=[5, 5], dpi=300)
    # plt.plot(loss[:], color='k')
    # plt.plot(val_loss[30:], color='r')
    # plt.yscale('log')
    # plt.show()
    # loss = history_mae.history['loss']
    # plt.figure(figsize=[5, 5], dpi=300)
    # plt.plot(loss[-100:], color='k')
    # plt.yscale('log')
    # plt.show()

    '''
    model_mae.i_model.load_weights("save_weights/26_04_2022 17_47/mae_loss0.01289516594260931.h5")
    # '''

    prediction_losses(model_mae,
                      frames, points,
                      min_c_training, max_c_training, min_c_labels, max_c_labels,
                      step)
    for i in range(0, 16):
        data_mae = creating_data_graph(frames, points, [i], 3, -1, step)
        data_mae[0] = data_normalisation(data_mae[0], min_c_training, max_c_training)
        data_mae[1] = data_normalisation(data_mae[1], min_c_labels, max_c_labels)
        prediction_gif(model_mae, data_mae, min_c_training, max_c_training, min_c_labels, max_c_labels, step,
                       name="pred_gif_plot" + str(i))

    # '''


def position_transform(x, y, dx, dy):
    shape = backend.shape(x)
    nx = backend.reshape([x + dx], shape=(shape[0], shape[1], 1))
    ny = backend.reshape([y + dy], shape=(shape[0], shape[1], 1))
    data = backend.concatenate([nx, ny], axis=2)
    return data


def prediction_gif(model_mae, initial_data, min_c, max_c, min_c_1, max_c_1, step, name="pred_gif"):
    prediction = np.array([initial_data[0][0]])
    image_array = []
    x_predictions = []
    y_predictions = []
    x_actual = []
    y_actual = []
    for f in range(len(initial_data[0])-5):
        fig = plt.Figure(figsize=[3, 3], dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        pred = model_mae(prediction).numpy()
        corr_x = data_unnormalisation(prediction[0, :, 0], min_c[:, :, 0], max_c[:, :, 0])
        corr_y = data_unnormalisation(prediction[0, :, 1], min_c[:, :, 1], max_c[:, :, 1])
        diff_x = data_unnormalisation(pred[0, :, 0], min_c_1[:, :, 0, 0], max_c_1[:, :, 0, 0])
        diff_y = data_unnormalisation(pred[0, :, 1], min_c_1[:, :, 0, 1], max_c_1[:, :, 0, 1])
        data_adjusted = position_transform(corr_x, corr_y, diff_x, diff_y).numpy()
        prediction = data_normalisation(data_adjusted, min_c, max_c)
        x = data_unnormalisation(initial_data[0][f * step + step, :, 0], min_c[:, :, 0], max_c[:, :, 0])
        y = data_unnormalisation(initial_data[0][f * step + step, :, 1], min_c[:, :, 1], max_c[:, :, 1])
        x_predictions.append(data_adjusted[0, :, 0])
        y_predictions.append(data_adjusted[0, :, 1])
        x_actual.append(x)
        y_actual.append(y)
        ax.scatter(y, x, s=0.5)
        ax.scatter(data_adjusted[0, :, 1], data_adjusted[0, :, 0], s=0.5)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.axhline(0)
        ax.axis('off')
        plt.show()
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_array.append(image)
    make_gif(image_array, name)
    return [x_predictions, y_predictions], [x_actual, y_actual]


def prediction_losses(model_mae, frames, points, min_c_mae, max_c_mae, min_c_mae_1, max_c_mae_1, step):
    plt.figure(dpi=200, figsize=[5, 5])
    loss_totals = []
    colors = cm.winter(np.linspace(0, 1, 16))
    for sim in range(16):
        data_mae = creating_data_graph(frames, points, [sim], 3, -1, step)
        copy_data = np.copy(data_mae[0])
        data_mae[0] = data_normalisation(data_mae[0], min_c_mae, max_c_mae)
        data_mae[1] = data_normalisation(data_mae[1], min_c_mae_1, max_c_mae_1)
        prediction = np.array([data_mae[0][0]])
        x_predictions = []
        y_predictions = []
        x_actual = []
        y_actual = []
        for f in range(int(len(data_mae[0]) / step) - 5):
            pred = model_mae(prediction).numpy()
            corr_x = data_unnormalisation(prediction[0, :, 0], min_c_mae[:, :, 0], max_c_mae[:, :, 0])
            corr_y = data_unnormalisation(prediction[0, :, 1], min_c_mae[:, :, 1], max_c_mae[:, :, 1])
            diff_x = data_unnormalisation(pred[0, :, 0], min_c_mae_1[:, :, 0, 0], max_c_mae_1[:, :, 0, 0])
            diff_y = data_unnormalisation(pred[0, :, 1], min_c_mae_1[:, :, 0, 1], max_c_mae_1[:, :, 0, 1])
            data_adjusted = position_transform(corr_x, corr_y, diff_x, diff_y)
            prediction = data_normalisation(data_adjusted, min_c_mae, max_c_mae)

            x = copy_data[f * step + step, :, 0]
            y = copy_data[f * step + step, :, 1]
            x_predictions.append(data_adjusted[0, :, 0])
            y_predictions.append(data_adjusted[0, :, 1])
            x_actual.append(x)
            y_actual.append(y)
        xy_predictions, xy_actual = [x_predictions, y_predictions], [x_actual, y_actual]
        loss_arr = []
        for i in range(len(xy_predictions[0])):
            xa = xy_actual[1][i]
            xp = xy_predictions[1][i]
            mse = losses.mean_squared_error(xa, xp).numpy()
            loss_arr.append(mse)
        plt.plot(loss_arr[:], label=sim, color=colors[sim])
        loss_totals.append(loss_arr)
    arr = pd.DataFrame(loss_totals).mean(axis=0).to_numpy()
    plt.plot(arr, color='k')
    plt.yscale('log')
    plt.ylim((pow(10, -9), 0.01))
    plt.legend()
    plt.show()


def make_gif(images, name):
    imageio.mimsave("{}.gif".format(name), images)


def velocity_calculation(data, time_step):
    data = np.array(data)
    velocity = [data[time_step] - data[0]]
    for i in range(1, len(data) - time_step):
        vel = (data[i + time_step] - data[i])
        velocity.append(vel)
    return np.array(velocity)


def load_data(points, frames, time_step, simulation_array, initial, final):
    training_data = []
    labels = []
    pbar = tqdm(total=len(simulation_array))
    for simulations in simulation_array:
        simulation = simulations
        pbar.update()
        data_names = glob.glob("training_data/new_xmin_Simulation_{}_points_{}/*".format(simulation, points))
        folder_length = len(data_names)
        data_array = []
        if final == -1:
            end_point = folder_length - 1
        else:
            end_point = final
        for file_num in range(initial, end_point - 5):
            data = np.load(
                "training_data/new_xmin_Simulation_{}_points_{}/data_{}.npy".format(simulation, points, file_num))
            data_array.append(data)
        xy_data = []
        for data in data_array:
            xy_array = []
            for i in range(points):
                xy = []
                for j in range(2):
                    xy.append(data[j][i])
                xy_array.append(xy)
            xy_data.append(xy_array)
        vel_data = velocity_calculation(xy_data, time_step)
        for i in range(0, len(xy_data) - frames * time_step - time_step):
            row = np.array(xy_data[i])
            if frames > 1:
                vel = []
                for j in range(frames):
                    index = i + j * time_step
                    vel.append(vel_data[index])
            else:
                vel = vel_data[i]
            labels.append(vel)
            training_data.append(row)
    pbar.close()
    return [np.array(training_data), np.array(labels)]


def creating_data_graph(frames, points, simulations_array, initial, final, step):
    data = load_data(points, frames, step, simulations_array, initial, final)
    if frames > 1:
        data[1] = np.transpose(data[1], axes=(0, 2, 1, 3))
    return data


def data_normalisation_constants(data):
    min_array = []
    max_array = []

    min_placeholder = np.min(data, axis=0)
    max_placeholder = np.max(data, axis=0)
    if len(np.shape(data)) == 5:
        min_array.append(min_placeholder[:, 1])
        max_array.append(max_placeholder[:, 1])
    else:
        min_array.append(min_placeholder)
        max_array.append(max_placeholder)

    return np.array(min_array), np.array(max_array)


def data_normalisation(data, min_array, max_array):
    u = 1
    l = -1
    try:
        for i in range(len(data[0])):
            data[:, i] = ((data[:, i] - min_array[i]) / (max_array[i] - min_array[i])) * (u - l) + l
    except:
        data = ((data - min_array) / (max_array - min_array)) * (u - l) + l
    return data


def data_unnormalisation(data, min_, max_):
    u = 1
    l = -1
    data = (data - l) * (max_ - min_) / (u - l) + min_
    return data


if __name__ == "__main__":
    main()
