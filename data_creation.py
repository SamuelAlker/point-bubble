import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio

BASE_SIZE = 540


def read_file(file_path):
    """Takes in a filepath and extracts a set of x and y coordinates of the bubble edge.
    Input:
        A string of the file path of a .dat file to read in
    Output:
        A pair of 1D  arrays of floats (x and y)
    """
    x = []
    y = []
    try:
        file = open(file_path, "r")
        main_data = False
        for line in file:
            # Excluding data that is irrelevant (the walls of the image)
            if "boundary4" in line:
                main_data = True
            if main_data and "ZONE" not in line:
                data_points = line.strip().split(" ")
                x.append(float(data_points[1]))
                y.append(float(data_points[0]))
        file.close()
    except IOError:
        print("File {} not found.".format(file_path))
        x = []
        y = []
    except ValueError:
        print("One of the lines in {} was unexpected.".format(file_path))
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


def make_lines(x, y, resolution):
    """Creates a series of interpolated points between raw bubble edge data points.
    Inputs:
        x: A 1D array of floats from raw data points
        y: A 1D array of floats from raw data points
        resolution: A float, representing how close the interpolated points should be.
    Outputs:
        filled_x: A 1D array of interpolated data points
        filled_y: A 1D array of interpolated data points
    """
    current_x = x[0]
    current_y = y[0]
    visited = [0]
    while len(visited) < len(x):
        checked = []
        values = []
        for i in range(0, len(x)):
            if i not in visited:
                checked.append(i)
                values.append(
                    (current_x - x[i]) ** 2 + (current_y - y[i]) ** 2
                )
        closest = min(values)
        smallest = checked[values.index(closest)]
        visited.append(smallest)
        current_x = x[smallest]
        current_y = y[smallest]

    new_x = []
    new_y = []
    for i in visited:
        new_x.append(x[i])
        new_y.append(y[i])

    filled_x = []
    filled_y = []

    for i in range(0, len(new_x)):
        current_x = float(new_x[i])
        current_y = float(new_y[i])
        if i + 1 != len(new_x):
            next_x = float(new_x[i + 1])
            next_y = float(new_y[i + 1])
        else:
            next_x = float(new_x[0])
            next_y = float(new_y[0])
        angle_to_next = np.arctan2(next_x - current_x, next_y - current_y)
        distance = np.sqrt((current_x - next_x) ** 2 + (current_y - next_y) ** 2)
        loops = 0
        while resolution * loops < distance:
            filled_x.append(current_x)
            filled_y.append(current_y)
            loops += 1
            current_x += resolution * np.sin(angle_to_next)
            current_y += resolution * np.cos(angle_to_next)
    filled_x = np.asarray(filled_x)
    filled_y = np.asarray(filled_y)

    return filled_x, filled_y


def convert_dat_files(variant_range, resolution=0.0001, inversions=[False]):
    """Converts all .dat files to numpy arrays, and saves them as .bmp files.
    These .bmp files are stored in Simulation_data_extrapolated/Simulation_X,
    where X is the reference number for the simulation.
    These aren't necessarily actual simulations, but can be variants of these 'base' simulations,
    where the physics remains constant.
    Input:
        variant_range:  An array of two floats, defining the [minimum, maximum]
                            amount to shift the original images in the x-axis. This range is inclusive.
        resolution: (default 0.0001) A float defining the distance between points when the raw data is interpolated.
    Output:
        Nothing
    """
    simulation_names = glob.glob("Simulation_data/*")
    simulation_index = 0
    for simulation in simulation_names:
        dat_files = glob.glob("{}/b*.dat".format(simulation))
        tracking_index = 0
        pbar = tqdm(total=len(dat_files))
        data_array = []
        for file in dat_files:
            pbar.update(1)
            # Extracting data
            x, y = read_file(file)
            # Finding the actual frame number
            step_number = int(file[file.find("s_") + 2:-4])
            # Converting to array
            x, y = make_lines(x, y, resolution)
            data_array.append([x, y, step_number])

        tracking_index += 1
        pbar.close()
        for inversion in inversions:
            for variant in variant_range:
                if inversion:
                    print("File {}, flipped, shifted {} is now Simulation_{}_{}_{}_{}".format(
                        simulation, variant, inversion, variant, resolution, simulation_index
                    ))
                else:
                    print("File {}, shifted {} is now Simulation_{}_{}_{}_{}".format(
                        simulation, variant, inversion, variant, resolution, simulation_index
                    ))
                # Making a directory for the images
                try:
                    os.mkdir("Simulation_data_extrapolated/Simulation_{}_{}_{}_{}".format(
                        inversion, variant, resolution, simulation_index
                    ))
                except OSError:
                    print("Folder already exists!")
                for row in data_array:
                    x = row[0]
                    y = row[1]
                    step_number = row[2]
                    # Saving to memory
                    data = np.array(
                        [((-1) ** (not inversion)) * x, y + (variant / BASE_SIZE)])  # numpy switch the x and y here!
                    np.save("Simulation_data_extrapolated/Simulation_{}_{}_{}_{}/data_{}".format(
                        inversion, variant, resolution, simulation_index, step_number
                    ), data)
                # Now the heavy lifting
        simulation_index += 1


def find_zero(x_array, y_array):
    x_copy = np.array(x_array)
    best_idx = 0
    for i in range(20):
        idx = (np.abs(x_copy - 0)).argmin()
        if y_array[idx] > y_array[best_idx]:
            best_idx = idx
        x_copy[idx] = 999
    return best_idx


def positive_values(x_array, y_array):
    for i in range(len(y_array)):
        if y_array[i] < 0:
            y_array[i] = 0
    return x_array, y_array


def plot_for_offset(file_num):
    # Data for plotting
    file = np.load("Simulation_data_extrapolated/Simulation_False_0_0.0001_0/data_" + str(file_num) + ".npy")

    idx = find_zero(file[1], file[0])
    print(file_num, idx)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(file[1], file[0])
    plt.axhline(file[0][idx], color="r")
    ax.grid()
    ax.set(xlabel='X', ylabel='x^{}'.format(file_num),
           title='Powers of x')

    # IMPORTANT ANIMATION CODE HERE
    # Used to keep the limits constant
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


def circumfrance(file):
    length = 0
    for i in range(len(file[0]) - 1):
        x_1 = file[0][i]
        y_1 = file[1][i]
        x_2 = file[0][i + 1]
        y_2 = file[1][i + 1]
        length += ((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) ** 0.5
    return length


def redefine_index(arrays, index):
    arr_length = len(arrays[0])
    new_arrays = [np.zeros(arr_length), np.zeros(arr_length)]
    for i in range(arr_length - 1):
        prime_index = index + i
        new_arrays[0][i] = arrays[0][prime_index]
        new_arrays[1][i] = arrays[1][prime_index]
        if index + i == arr_length - 1:
            index = -i
    return new_arrays


def circ_array(arrays):
    arr_length = len(arrays[0])
    new_array = np.zeros(arr_length)
    length = 0
    for i in range(arr_length - 1):
        x_1 = arrays[0][i]
        y_1 = arrays[1][i]
        x_2 = arrays[0][i + 1]
        y_2 = arrays[1][i + 1]
        length += ((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2) ** 0.5
        new_array[i + 1] = length
    return new_array


def border_data(point_num, simulation_num, file_num):
    # data_names = glob.glob("Simulation_data_extrapolated/Simulation_False_0_0.0001_0/*")
    file = np.load("Simulation_data_extrapolated/Simulation_False_0_0.0001_" + str(simulation_num) + "/data_" + str(
        file_num) + ".npy")
    final_array = [np.zeros(point_num), np.zeros(point_num)]
    idx = find_zero(file[1], file[0])
    circ = circumfrance(file)
    length = circ / point_num
    data = redefine_index(file, idx)
    # data[1][0] = 0
    circ_data = circ_array(data)
    for i in range(point_num):
        idx = (np.abs(circ_data - length * i)).argmin()
        final_array[0][i] = data[0][idx]
        final_array[1][i] = data[1][idx]
    return final_array


def save_border_data(point_num, simulation):
    data_names = glob.glob("Simulation_data_extrapolated/Simulation_False_0_0.0001_{}/*".format(str(simulation)))
    folder_length = len(data_names)
    try:
        os.mkdir("training_data/Simulation_{}_points_{}/".format(simulation, point_num))
    except OSError:
        print("Folder already exists!")
    pbar = tqdm(total=folder_length - 3)
    for i in range(3, folder_length):
        pbar.update()
        final_data = border_data(point_num, simulation, i)
        np.save("training_data/Simulation_{}_points_{}/data_{}".format(simulation, point_num, i), final_data)
    pbar.close()


def make_gif(images, name):
    imageio.mimsave("{}.gif".format(name), images)


def main():
    # convert_dat_files([0], 0.0001)
    for i in range(1, 11):
        save_border_data(1000, i)
    for i in range(1, 11):
        save_border_data(100, i)

    # simulation = 0
    # points = 100
    # plot_gif(points, simulation)

    # kwargs_write = {'fps': 0.2 , 'quantizer': 'nq'}
    # imageio.mimsave('./powers.gif', [plot_for_offset(i) for i in range(900, 970)], fps=0.2)

    # plot_circ_graphs()


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


def plot_circ_graphs():
    for i in range(700, 960, 20):
        final_data = border_data(100, 0, i)
        file = np.load("Simulation_data_extrapolated/Simulation_False_0_0.0001_0/data_" + str(i) + ".npy")
        plt.figure(figsize=[5, 5])
        plt.axvline(0, color="r")
        plt.plot(file[1], file[0])
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.scatter(final_data[1], final_data[0], s=10.5, color="g")
        plt.show()


if __name__ == "__main__":
    main()
