import numpy as np
import glob
from tqdm import tqdm
from tensorflow.keras import layers, models, initializers, activations, losses, metrics, optimizers, Model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

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

def model_1(activation, optimizer, frames=4, size=100, kernal_size=32):
    initializer = initializers.HeNormal()
    model = models.Sequential()
    model.add(layers.Conv2D(kernal_size, (frames, 3), kernel_initializer=initializer, activation=activation, input_shape=(frames, size, 2)))
    model.add(layers.Reshape((size-2, kernal_size)))
    model.add(layers.Conv1D(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(128, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(128, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.Conv1DTranspose(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model.add(layers.UpSampling1D(3))
    model.add(layers.Conv1DTranspose(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model.add(layers.UpSampling1D(3))
    model.add(layers.Conv1DTranspose(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv1DTranspose(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv1DTranspose(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model.add(layers.Conv1D(2, 3, activation=activation, kernel_initializer=initializer))

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    print(model.summary())
    short_model_summary = "\n".join(stringlist)
    model.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=[metrics.MeanSquaredError(name="MSE")])
    return model

def model_2(activation, optimizer, frames=4, size=100, kernal_size=32):
    initializer = initializers.HeNormal()
    model = models.Sequential()
    model.add(layers.Conv2D(kernal_size, (frames, 3), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, 2)))
    model.add(layers.Reshape((size - 2, kernal_size)))
    model.add(layers.Conv1D(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(128, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(128, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.Flatten())
    model.add(layers.Dense(400, activation=activation))
    model.add(layers.Dense(200, activation=activation))
    model.add(layers.Reshape((100, 2)))
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    print(model.summary())
    short_model_summary = "\n".join(stringlist)
    model.compile(optimizer=optimizer, loss=losses.MeanSquaredLogarithmicError(), metrics=[metrics.MeanSquaredError(name="MSE")])
    return model


def model_3(activation, optimizer, frames=4, size=100, kernal_size=32):
    initializer = initializers.HeNormal()
    model = models.Sequential()
    model.add(layers.Conv2D(kernal_size, (frames, 3), kernel_initializer=initializer, activation=activation,
                            input_shape=(frames, size, 2)))
    model.add(layers.Reshape((size - 2, kernal_size)))
    model.add(layers.Conv1D(kernal_size, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(128, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(128, 3, activation=activation, kernel_initializer=initializer))
    model = inception_cell(model, activation, 2, 32)
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation=activation))
    model.add(layers.Dense(20, activation=activation))
    model.add(layers.Dense(200, activation=activations.linear))
    model.add(layers.Reshape((100, 2)))
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    print(model.summary())
    short_model_summary = "\n".join(stringlist)
    model.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=[metrics.MeanSquaredError(name="MSE")])
    return model



def load_data(points, frames):
    training_data = []
    labels = []
    pbar = tqdm(total=11)
    for simulation in range(11):
        pbar.update()
        data_names = glob.glob("training_data/Simulation_{}_points_{}/*".format(simulation, points))
        folder_length = len(data_names)
        data_array = []
        for file_num in range(3, folder_length+1):
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
        for i in range(folder_length-frames-2):
            single = []
            for j in range(frames):
                single.append(xy_data[i+j])
            training_data.append(single)
            labels.append(xy_data[i+frames])
    pbar.close()
    return [np.array(training_data), np.array(labels)]


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
    plt.scatter(*zip(*prediction[0]))
    plt.Figure(figsize=[5, 5], dpi=300)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    plt.clf()
    prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 2))
    image_array = []
    for i in range(gif_length):
        fig = plt.Figure(figsize=[5, 5], dpi=300)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        prediction = model(prediction)
        ax.scatter(*zip(*prediction[0]))

        # ax.scatter(prediction[0])
        prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 2))
        ax.axvline(0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_array.append(image)
    make_gif(image_array, "pred_gif")


def main():
    print("Running Training")
    activation = activations.relu
    optimizer = optimizers.Adam(learning_rate=0.0001)
    frames = 1
    points = 100
    kernal_size = 32
    data = load_data(points, frames)
    model = model_3(activation, optimizer, frames, points, kernal_size)
    history = model.fit(data[0], data[1], epochs=30, shuffle=True)
    prediction_gif(model, data[0][0], 10)






if __name__ == "__main__":
    main()