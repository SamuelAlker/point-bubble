import ast

import numpy as np
import glob
from tqdm import tqdm
from tensorflow.keras import layers, initializers, activations, losses, metrics, optimizers, Model, callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
import models
from datetime import datetime
import os


def inception_cell(model, activation, axis, kernal_size):
    shape = model.output_shape
    li = list(shape)
    li.pop(0)
    shape = tuple(li)
    input_tower = layers.Input(shape=shape)

    tower_1 = layers.Conv1D(kernal_size, 1, padding='same', activation=activation)(
        input_tower)

    tower_2 = layers.Conv1D(kernal_size, 1, padding='same', activation=activation)(
        input_tower)
    tower_2 = layers.Conv1D(kernal_size, 3, padding='same', activation=activation)(
        tower_2)

    tower_3 = layers.Conv1D(kernal_size, 1, padding='same', activation=activation)(
        input_tower)
    tower_3 = layers.Conv1D(kernal_size, 5, padding='same', activation=activation)(
        tower_3)

    # tower_4 = layers.MaxPooling2D((3, 3), strides=1)(input_tower)
    tower_4 = layers.Conv1D(kernal_size, 3, padding='same', activation=activation)(
        input_tower)

    merged = layers.concatenate([tower_1, tower_2, tower_3, tower_4], axis=axis)

    model.add(Model(input_tower, merged))
    return model


def make_gif(images, name):
    imageio.mimsave("{}.gif".format(name), images)


def plot_gif(points, simulation):
    data_names = glob.glob("training_data/Simulation_{}_points_{}/*".format(simulation, points))
    folder_length = len(data_names)
    image_array = []
    for i in range(3, folder_length, 5):
        fig = plt.Figure(figsize=[5, 5], dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        data = np.load("training_data/Simulation_{}_points_{}/data_{}.npy".format(simulation, points, i))
        ax.scatter(data[1], data[0])
        ax.axvline(0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_array.append(image)
    make_gif(image_array, "scatter2")


def prediction_gif(model, initial_data, gif_length):
    prediction = np.array([initial_data[0][10]])
    # plt.scatter(*zip(*prediction[0, :, 0:2]))
    # plt.Figure(figsize=[5, 5], dpi=300)
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # plt.show()
    # plt.clf()
    # prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 4))
    image_array = []
    for f in range(gif_length):
        fig = plt.Figure(figsize=[5, 5], dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        pred = model.call_2(prediction).numpy()
        # prediction = prediction[:, :, :, :-1]
        data_adjusted = position_transform([prediction[0, :, -1, 0], prediction[0, :, -1, 1]], [pred[0, :, 0], pred[0, :, 1]], 100)
        prediction[0, :, 0] = prediction[0, :, 1]
        for i in range(len(prediction[0])):
            prediction[0, i, 1, 0] = data_adjusted[0][i]
            prediction[0, i, 1, 1] = data_adjusted[1][i]
            prediction[0, i, 1, 2] = pred[0, i, 0]
            prediction[0, i, 1, 3] = pred[0, i, 1]

        prediction = distance_conversion([prediction, [0]], pbar_toggle=False)
        ax.scatter(data_adjusted[0], data_adjusted[1])
        x = initial_data[0][15+f*5, :, -1, 0]
        y = initial_data[0][15+f*5, :, -1, 1]
        ax.scatter(x, y)

        # ax.scatter(prediction[0])
        # prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 2))
        ax.axvline(0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_array.append(image)
    make_gif(image_array, "pred_gif")


def velocity_calculation(data, scaling, time_step):
    data = np.array(data)
    velocity = [data[time_step] - data[0]]
    for i in range(1, len(data) - time_step):
        vel = (data[i + time_step] - data[i])*scaling
        velocity.append(vel)
    return np.array(velocity)


def load_data(points, frames, time_step, scaling):
    training_data = []
    labels = []
    pbar = tqdm(total=10)
    for simulations in range(10):
        simulation=simulations
        pbar.update()
        data_names = glob.glob("training_data/xmin_Simulation_{}_points_{}/*".format(simulation, points))
        folder_length = len(data_names)
        data_array = []
        for file_num in range(3, folder_length + 1):
            data = np.load("training_data/xmin_Simulation_{}_points_{}/data_{}.npy".format(simulation, points, file_num))
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
        vel_data = velocity_calculation(xy_data, scaling, time_step)
        for i in range(0, len(xy_data) - frames * time_step - time_step):
            single = []
            for j in range(0, frames):
                row = np.array(xy_data[i + j * time_step + time_step])
                row = np.append(row, vel_data[i + j * time_step], axis=1)
                single.append(row)
                # single.append(vel_data[i + j])
            vel = vel_data[i + frames * time_step]
            # if np.max(vel) < 20:
            #     if np.max(single) < 20:
            training_data.append(single)
            labels.append(vel)
    pbar.close()
    return [np.array(training_data), np.array(labels)]


def position_transform(data, change, scaling):
    data[0] = change[0] / scaling + data[0]
    data[1] = change[1] / scaling + data[1]
    return data


def distance_conversion(data, pbar_toggle=True, save_files=False):
    shape = np.shape(data[0])
    new_data = np.zeros((shape[0], shape[1], shape[2], shape[3] + shape[1]))
    if save_files:
        np.set_printoptions(threshold=np.inf)
        try:
            os.mkdir("Training_Strings/")
            os.mkdir("Label_Strings/")
        except OSError:
            print("Folders already exists!")
    if pbar_toggle:
        pbar = tqdm(total=shape[0])
    for i in range(len(data[0])):
        row = np.zeros((shape[1], shape[2], shape[3] + shape[1]))
        if pbar_toggle:
            pbar.update(1)
        for col_prime in range(len(data[0][0])):
            row[col_prime, :, :4] = data[0][i, col_prime, :, :4]
            for col_sub in range(len(data[0][0])):
                for frame in range(len(data[0][0, 0])):
                    distance = ((data[0][i, col_prime, frame, 0] - data[0][i, col_sub, frame, 0]) ** 2 + (data[0][i, col_prime, frame, 1] - data[0][i, col_sub, frame, 1]) ** 2) ** 0.5
                    row[col_prime, frame, col_sub+4] = distance
        new_data[i] = row
        if save_files:
            training_data = np.array2string(row, separator=',')
            labels = np.array2string(data[1][i], separator=',')
            training_file = open("Training_Strings/data_{}.txt".format(i), "w")
            training_file.write(training_data)
            training_file.close()
            labels_file = open("Label_Strings/label_{}.txt".format(i), "w")
            labels_file.write(labels)
            labels_file.close()

    if pbar_toggle:
        pbar.close()
    return new_data


def main():
    print("Running Training")
    activation = activations.tanh
    optimizer = optimizers.Adam(learning_rate=0.001)
    frames = 2
    points = 100
    kernal_size = 32
    scaling = 100

    # a1 = np.array([[[1],[1],[1]], [[1],[1],[1]]])
    # a2 = np.array([[[1], [1], [1]], [[1], [1], [1]]])

    today = datetime.today()
    dt_string = today.strftime("%d_%m_%Y_%H_%M")
    directory = "saved_weights/" + dt_string

    # datas = tf.data.Dataset.from_tensor_slices((a1, a2))



    save_data = 1
    if save_data:
        data = load_data(points, frames, 5, scaling)
        data = [data[0][:], data[1][:]]
        data[0] = np.transpose(data[0], axes=(0, 2, 1, 3))
        data[0] = distance_conversion(data, save_files=False)
    else:
        training_names = glob.glob("Training_Strings/*")
        label_names = glob.glob("Label_Strings/*")
        folder_length = len(training_names)
        labels_array = []
        training_array = []
        pbar = tqdm(total=folder_length)
        for i in range(folder_length):
            pbar.update(1)
            training_file = open(training_names[i], "r")
            labels_file = open(label_names[i], "r")
            training_data = training_file.read()
            training_array.append(np.array(ast.literal_eval(training_data)))
            labels = labels_file.read()
            labels_array.append(np.array(ast.literal_eval(labels)))
            training_file.close()
            labels_file.close()
        pbar.close()
        data = [np.array(training_array), np.array(labels_array)]


    xx = data[1][:, 0, 0]
    print(np.argmax(xx))
    yy = data[1][:, 0, 1]
    plt.plot(xx)
    plt.show()
    plt.plot(yy)
    plt.show()

    datas = tf.data.Dataset.from_tensor_slices((data[0][:], data[1][:])).shuffle(buffer_size=10000).batch(1)

    # model = models.resnet(activation, optimizer, 100, frames)[0]

    model = models.graph_network(frames, activation, 5, 10, 2, optimizer)
    test_data = np.array([data[0][10]])

    def step_decay(epoch):
        int_rate = 0.001
        return np.exp(np.log(0.1)/20)**epoch * int_rate

    train = 1
    lr = callbacks.LearningRateScheduler(step_decay)
    if train:
        history = model.fit(datas, epochs=5, callbacks=[lr])
        tf.config.run_functions_eagerly(True)
        pred = model(test_data)
        model.save_weights(directory + '.h5')
    else:
        list_of_files = glob.glob('saved_weights/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
        model = models.graph_network(frames, activation, 5, 10, 2, optimizer)
        pred = model(test_data)
        model.load_weights(latest_file)

    x = data[0][10, :, -1, 0]
    y = data[0][10, :, -1, 1]
    dx = data[1][10, :, 0]
    dy = data[1][10, :, 1]
    plt.scatter(x, y, color='r')
    data_adjusted = position_transform([x, y], [dx, dy], scaling)
    plt.scatter(data_adjusted[0], data_adjusted[1], color='g')
    # plt.show()
    pred = model.call_2(test_data).numpy()
    data_adjusted = position_transform([x, y], [pred[0, :, 0], pred[0, :, 1]], scaling)
    # plt.scatter(x, y, color='r')
    plt.scatter(data_adjusted[0], data_adjusted[1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    dx_p = pred[0, :, 0]
    dy_p = pred[0, :, 1]
    plt.scatter(dx_p, dy_p)
    plt.scatter(dx, dy, color='r')
    plt.show()

    prediction_gif(model, data, 20)


if __name__ == "__main__":
    main()
