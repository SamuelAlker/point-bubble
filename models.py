from tensorflow.keras import layers, Model, losses, metrics, regularizers, activations


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
    x_input = layers.Input(shape=(frames, input_shape, 2), name="input")
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
    print(model.summary())

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    sum = "\n".join(stringlist)

    # model = Custom_Model.CustomModel(model)
    # model.compile(optimizer=optimizer)

    metric = metrics.MeanAbsoluteError(name="MAE")
    model.compile(optimizer=optimizer, loss=losses.MeanSquaredError(), metrics=[metric], )
    return [model, sum]