import numpy as np
import glob
from tqdm import tqdm
from tensorflow.keras import layers, initializers, activations, losses, metrics, optimizers, Model
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
import models
from datetime import datetime



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
    prediction = initial_data
    plt.scatter(*zip(*prediction[0, :, 0:2]))
    plt.Figure(figsize=[5, 5], dpi=300)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    plt.clf()
    prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 4))
    image_array = []
    for i in range(gif_length):
        fig = plt.Figure(figsize=[5, 5], dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        n_prediction = model(prediction)
        prediction[0,:,:,2:] = n_prediction.numpy()
        prediction[0, :, :, :2] = prediction[0, :, :, :2] + (n_prediction.numpy()/100)
        ax.scatter(*zip(*prediction[0, 0, :, 0:2]))

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
    for i in range(1, len(data)-time_step):
        velocity.append((data[i+time_step]-data[i])*scaling)
    return np.array(velocity)


def load_data(points, frames, time_step):
    training_data = []
    labels = []
    pbar = tqdm(total=2)
    for simulation in range(2):
        pbar.update()
        data_names = glob.glob("training_data/Simulation_{}_points_{}/*".format(simulation, points))
        folder_length = len(data_names)
        data_array = []
        for file_num in range(3, folder_length + 1):
            data = np.load("training_data/Simulation_{}_points_{}/data_{}.npy".format(simulation, points, file_num))
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
        vel_data = velocity_calculation(xy_data, 100, time_step)
        for i in range(0, len(xy_data) - frames*time_step - time_step):
            single = []
            for j in range(0, frames):
                row = np.array(xy_data[i + j*time_step + time_step])
                row = np.append(row, vel_data[i+j*time_step], axis=1)
                single.append(row)
                # single.append(vel_data[i + j])
            training_data.append(single)
            labels.append(vel_data[i + frames*time_step])
    pbar.close()
    return [np.array(training_data), np.array(labels)]


def main():
    print("Running Training")
    activation = activations.relu
    optimizer = optimizers.Adam()
    frames = 2
    points = 100
    kernal_size = 32

    # a1 = np.array([[[1],[1],[1]], [[1],[1],[1]]])
    # a2 = np.array([[[1], [1], [1]], [[1], [1], [1]]])

    # datas = tf.data.Dataset.from_tensor_slices((a1, a2))

    data = load_data(points, frames, 5)
    data[0] = np.transpose(data[0], axes=(0, 2, 1, 3))
    datas = tf.data.Dataset.from_tensor_slices((data[0], data[1])).batch(32)

    # model = models.resnet(activation, optimizer, 100, frames)[0]
    model = models.graph_network(frames, activation, 4, 4, 2)

    history = model.fit(datas, epochs=5, shuffle=True)
    # model = tf.keras.models.load_model("saved_models/10_03_2022_16_25")

    today = datetime.today()
    # dt_string = today.strftime("%d_%m_%Y_%H_%M")
    # directory = "saved_models/" + dt_string
    # model.save(directory)
    # prediction_gif(model, data[0][400], 10)


if __name__ == "__main__":
    main()
