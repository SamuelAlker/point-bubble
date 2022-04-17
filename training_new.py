import ast
import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import glob
from tqdm import tqdm
from tensorflow.keras import layers, initializers, activations, losses, metrics, optimizers, Model, callbacks, backend
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from keras.engine import data_adapter
from datetime import datetime


def residual_cell(x, activation, layer_size=2, size=10):
    x_skip = x
    for i in range(layer_size):
        x = layers.Dense(size, activation=activation)(x)
    input_size = x_skip.get_shape()[1]
    x = layers.Dense(input_size, activation=activation)(x)
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activation=activation)(x)
    return x


def dense_network(input_number, frames, points, activation, optimiser, input_nodes, nodes, layer_num, cell_count):
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
    #model.compile(optimizer=optimiser)
    # print(model.summary())
    #model = CustomModel(x_input, x)
    model.compile(optimizer=optimiser, loss=losses.mean_squared_error)
    return model


class CustomModel(Model):
    loss_tracker = metrics.Mean(name="loss")

    def train_step(self, data):
        input_data, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_1:
            tape_1.watch(self.trainable_variables)
            y_pred = self(input_data, training=True)
            loss_tot = losses.mean_squared_error(y, y_pred)
        self.optimizer.minimize(loss_tot, [self.trainable_variables], tape=tape_1)
        self.loss_tracker.update_state(loss_tot)
        # self.MAE_tracker.update_state(y, y_pred)
        loss = self.loss_tracker.result()
        
        # MAE = self.MAE_tracker.result()
        return {"loss": loss, "loss_tot": loss_tot}

    def test_step(self, data):
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        self.loss_tracker.reset_states()
        # y_pred = self(x, training=False)
        # self.mse.update_state(y, y_pred)
        # mse_result = self.mse.result()
        return {"MSE": 1}
    

def hyperparameter_explore(lr, datas, epochs, test_epochs, check_iteration):
    # tf.config.run_functions_eagerly(True)
    today = datetime.today()
    directory = today.strftime("%d_%m_%Y_%H_%M")

    input_nodes = 30
    input_nodes_adjust = 5
    best_input_nodes = input_nodes

    nodes = 30
    nodes_adjust = 5
    best_nodes = nodes

    layer_number = 2
    layer_number_adjust = 1
    best_layer_number = layer_number

    cells = 4
    cells_adjust = 1
    best_cells = cells
    
    backend.clear_session()

    activation = activations.swish
    optimizer = optimizers.Adam(learning_rate=0.001)
    save_model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes, layer_number, cells)
    print("Initial Save Model Training")
    history = save_model.fit(datas[0], datas[1], epochs=test_epochs, callbacks=[lr], verbose=2, validation_split=0.01)
    previous_loss = history.history["loss"][-1]
    iteration = 1
    input_loss = 999
    nodes_loss = 999
    layers_loss = 999
    cells_loss = 999
    
    while True:
        input_placehold = 0
        node_placehold = 0
        layer_placehold = 0
        cell_placehold = 0
        
        pbar = tqdm(total=4)
        
        backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()
        activation = activations.swish
        optimizer = optimizers.Adam(learning_rate=0.001)
        model = dense_network(100, 2, 4, activation, optimizer, input_nodes+input_nodes_adjust, nodes, layer_number, cells)
        history_input_nodes = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
        
        if input_nodes > input_nodes_adjust:
            backend.clear_session()
            gc.collect()
            tf.compat.v1.reset_default_graph()
            activation = activations.swish
            optimizer = optimizers.Adam(learning_rate=0.001)
            del model
            model = dense_network(100, 2, 4, activation, optimizer, input_nodes-input_nodes_adjust, nodes, layer_number, cells)
            history_input_nodes_negative = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
            if history_input_nodes.history["loss"][-1] < history_input_nodes_negative.history["loss"][-1]:
                input_placehold = input_nodes + input_nodes_adjust
                input_loss = history_input_nodes.history["loss"][-1]
                
            else:
                input_placehold = input_nodes - input_nodes_adjust
                input_loss = history_input_nodes_negative.history["loss"][-1]
        else:
            if history_input_nodes.history["loss"][-1] < input_loss:
                input_placehold = input_nodes + input_nodes_adjust
                input_loss = history_input_nodes.history["loss"][-1]
                

        pbar.update(1)
        backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()
        del model
        activation = activations.swish
        optimizer = optimizers.Adam(learning_rate=0.001)
        model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes + nodes_adjust, layer_number, cells)
        history_nodes = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
        if nodes > nodes_adjust:
            backend.clear_session()
            gc.collect()
            tf.compat.v1.reset_default_graph()
            del model
            activation = activations.swish
            optimizer = optimizers.Adam(learning_rate=0.001)
            model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes - nodes_adjust, layer_number, cells)
            history_nodes_negative = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
            if history_nodes.history["loss"][-1] < history_nodes_negative.history["loss"][-1]:
                node_placehold = nodes + nodes_adjust
                nodes_loss = history_nodes.history["loss"][-1]
            else:
                node_placehold = nodes - nodes_adjust
                nodes_loss = history_nodes_negative.history["loss"][-1]
        else:
            if history_nodes.history["loss"][-1] < nodes_loss:
                node_placehold = nodes + nodes_adjust
                nodes_loss = history_nodes.history["loss"][-1]


        pbar.update(1)
        backend.clear_session()
        gc.collect()
        del model
        tf.compat.v1.reset_default_graph()
        activation = activations.swish
        optimizer = optimizers.Adam(learning_rate=0.001)
        model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes, layer_number + layer_number_adjust, cells)
        history_layer = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
        if layer_number > layer_number_adjust+0.1:
            backend.clear_session()
            gc.collect()
            del model
            tf.compat.v1.reset_default_graph()
            activation = activations.swish
            optimizer = optimizers.Adam(learning_rate=0.001)
            model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes, layer_number - layer_number_adjust, cells)
            history_layer_number_negative = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
            if history_layer.history["loss"][-1] < history_layer_number_negative.history["loss"][-1]:
                layer_placehold = layer_number + layer_number_adjust
                layers_loss = history_layer.history["loss"][-1]
                if layer_number == 0:
                    layer_number == 1
            else:
                layer_placehold = layer_number - layer_number_adjust
                layers_loss = history_layer_number_negative.history["loss"][-1]
                if layer_placehold == 0:
                    layer_placehold == 1
        else:
            if history_layer.history["loss"][-1] < layers_loss:
                layer_placehold = layer_number + layer_number_adjust
                layers_loss = history_layer.history["loss"][-1]
                if layer_number == 0:
                    layer_number == 1
        
        if layer_number == 0:
            layer_number == 1
        pbar.update(1)
        backend.clear_session()
        gc.collect()
        del model
        tf.compat.v1.reset_default_graph()
        activation = activations.swish
        optimizer = optimizers.Adam(learning_rate=0.001)
        model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes, layer_number, cells + cells_adjust)
        history_cells = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
        if cells > cells_adjust:
            backend.clear_session()
            gc.collect()
            del model
            tf.compat.v1.reset_default_graph()
            activation = activations.swish
            optimizer = optimizers.Adam(learning_rate=0.001)
            backend.clear_session()
            gc.collect()

            model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes, layer_number, cells - cells_adjust)
            history_cells_negative = model.fit(datas[0], datas[1], epochs=epochs, callbacks=[lr], verbose=0, shuffle=True, batch_size=32, validation_split=0.01)
            if history_cells.history["loss"][-1] < history_cells_negative.history["loss"][-1]:
                cell_placehold = cells + cells_adjust
                cells_loss = history_cells.history["loss"][-1]
            else:
                cell_placehold = cells + cells_adjust
                cells_loss = history_cells_negative.history["loss"][-1]
        else:
            if history_cells.history["loss"][-1] < cells_loss:
                cell_placehold = cells + cells_adjust
                cells_loss = history_cells.history["loss"][-1]
        pbar.update(1)
        pbar.close()
        min_loss = input_loss
        reduce = 0
        if nodes_loss < min_loss:
            min_loss = nodes_loss
            reduce = 1
        if layers_loss < min_loss:
            min_loss = layers_loss
            reduce = 2
        if cells_loss < min_loss:
            reduce = 3
        
        reduce=5
        
        if reduce == 0:
            input_nodes = input_placehold
        if reduce == 0:
            nodes = node_placehold
        if reduce == 2:
            layer_number = layer_number
        if reduce == 3:
            cells = cells
        
        
        input_node_string = "Input Node Number: " + str(input_nodes) + ", Loss = " + str(input_loss)
        node_string = "Node Number: " + str(nodes) + ", Loss = " + str(nodes_loss)
        layer_string = "Layer Number: " + str(layer_number) + ", Loss = " + str(layers_loss)
        cell_string = "Cell Number: " + str(cells) + ", Loss = " + str(cells_loss)
        print("#---------------------------#")
        print("Iteration", iteration)
        print(input_node_string)
        print(node_string)
        print(layer_string)
        print(cell_string)

        iteration += 1
        if iteration % check_iteration == 0:
            print("Testing Best Model")
            
            backend.clear_session()
            gc.collect()
            activation = activations.swish
            optimizer = optimizers.Adam(learning_rate=0.001)
            del model
            model = dense_network(100, 2, 4, activation, optimizer, input_nodes, nodes, layer_number, cells)
            history = model.fit(datas[0], datas[1], epochs=test_epochs, callbacks=[lr], verbose=0)
            loss = history.history["loss"][-1]
            print("Testing Loss = {}, Previous Loss = {}".format(loss, previous_loss))
            if loss < previous_loss:
                previous_loss = loss
                save_model = model
                best_input_nodes = input_nodes
                best_nodes = nodes
                best_layer_number = layer_number
                best_cells = cells
                del model
            else:
                break
    try:
        os.mkdir("Hyperparameter_Explore")
    except OSError:
        print("Hyperparameter_Explore folder already exists!")
    os.mkdir("Hyperparameter_Explore/{}".format(directory))
    training_file = open("Hyperparameter_Explore/{}/loss_{}_parameters".format(directory, previous_loss), "w")
    input_node_string = "Input Node Number: " + str(best_input_nodes)
    node_string = "Node Number: " + str(best_nodes)
    layer_string = "Layer Number: " + str(best_layer_number)
    cell_string = "Cell Number: " + str(best_cells)
    training_file.write(input_node_string)
    training_file.write(node_string)
    training_file.write(layer_string)
    training_file.write(cell_string)
    training_file.close()
    save_model.save_weights('Hyperparameter_Explore/{}/weights.h5'.format(directory))
    return save_model


def main():
    print("Running Training")
    activation = activations.swish
    optimizer = optimizers.Adam(learning_rate=0.001)
    frames = 2
    points = 100
    kernal_size = 32
    scaling = 100

    data = creating_data_graph(frames, points, scaling)

    #datas = tf.data.Dataset.from_tensor_slices((data[0][:], data[1][:])).shuffle(buffer_size=10000).batch(batch_size=32)


    def step_decay(epoch):
        int_rate = 0.001
        return np.exp(np.log(0.1)/100)**epoch * int_rate
    
    def step_decay_2(epoch):
        int_rate = 0.001
        return np.exp(epoch*np.log(0.1)/120) * (1+0.3*np.sin((2*3.14*epoch)/50)) * int_rate
    
    def step_decay_3(epoch):
        int_rate = 0.001
        return max(int_rate * 0.1 ** int(epoch/100), 1e-6)
        
            

    lr = callbacks.LearningRateScheduler(step_decay_3)
    parameter_epochs = 3
    testing_epochs = 5
    iteration_check = 50
    model = hyperparameter_explore(lr, data, parameter_epochs, testing_epochs, iteration_check)
    
    test_data = np.array([data[0][10]])
    
    '''
    xx = data[1][:, 0, 0]
    print(np.argmax(xx))
    yy = data[1][:, 0, 1]
    plt.plot(xx[:])
    plt.show()
    plt.plot(yy[:])
    plt.show()
    #'''
    
    #list_of_files = glob.glob('saved_weights/*')
    #latest_file = max(list_of_files, key=os.path.getctime)
    #print(latest_file)
    model = dense_network(100, 2, 4, activation, optimizer, 130, 65, 7, 23)
    pred = model(test_data)
    #model.load_weights("Hyperparameter_Explore\\ab\\weights.h5")
    print(model.summary())
    history = model.fit(data[0], data[1], epochs=250, callbacks=[lr], verbose=1, shuffle=True)

    
    test_data = np.array([data[0][10]])
    dx = data[1][10, :, 0]
    dy = data[1][10, :, 1]
    pred = model(test_data).numpy()
    dx_p = pred[0, :, 0]
    dy_p = pred[0, :, 1]
    plt.scatter(dx_p, dy_p)
    plt.scatter(dx, dy, color='r')
    plt.show()

    prediction_gif(model, data, 200)
    
    
def load_data(points, frames, time_step, scaling):
    training_data = []
    labels = []
    pbar = tqdm(total=9)
    for simulations in range(9):
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


def velocity_calculation(data, scaling, time_step):
    data = np.array(data)
    velocity = [data[time_step] - data[0]]
    for i in range(1, len(data) - time_step):
        vel = (data[i + time_step] - data[i])*scaling
        velocity.append(vel)
    return np.array(velocity)


def position_transform(data, change, scaling):
    data[0] = change[0] / scaling + data[0]
    data[1] = change[1] / scaling + data[1]
    return data


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
        fig = plt.Figure(figsize=[3, 3], dpi=200)
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        pred = model(prediction).numpy()
        # prediction = prediction[:, :, :, :-1]
        data_adjusted = position_transform([prediction[0, :, -1, 0], prediction[0, :, -1, 1]], [pred[0, :, 0], pred[0, :, 1]], 100)
        prediction[0, :, 0] = prediction[0, :, 1]
        for i in range(len(prediction[0])):
            prediction[0, i, 1, 0] = data_adjusted[0][i]
            prediction[0, i, 1, 1] = data_adjusted[1][i]
            prediction[0, i, 1, 2] = pred[0, i, 0]
            prediction[0, i, 1, 3] = pred[0, i, 1]

        # prediction = distance_conversion([prediction, [0]], pbar_toggle=False)
        x = initial_data[0][15+f*5, :, -1, 0]
        y = initial_data[0][15+f*5, :, -1, 1]
        ax.scatter(y, x, s=0.5)
        ax.scatter(data_adjusted[1], data_adjusted[0], s=0.5)

        # ax.scatter(prediction[0])
        # prediction = np.reshape(prediction, newshape=(1, 1, np.shape(prediction)[1], 2))
        ax.axvline(0)
        ax.axhline(0)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.axis('off')
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_array.append(image)
    make_gif(image_array, "pred_gif")


def make_gif(images, name):
    imageio.mimsave("{}.gif".format(name), images)


def creating_data_graph(frames, points, scaling):
    save_data = 1
    if save_data:
        data = load_data(points, frames, 5, scaling)
        data = [data[0][:], data[1][:]]
        data[0] = np.transpose(data[0], axes=(0, 2, 1, 3))
        # data[0] = distance_conversion(data, save_files=False)
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
    return data


if __name__ == "__main__":
    main()