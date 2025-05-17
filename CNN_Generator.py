import keras


def generate_Model(inputShape,filter_size,pool_size,pool_stride):
    insert = keras.layers.Input(shape=(inputShape[0],inputShape[1],inputShape[2]))
    output=0
    # TODO: layer size Berechnen
    # Conv Layer 1
    output = keras.layers.Conv2D(32,kernel_size=filter_size,activation="relu")(insert)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    #Conv Layer 2
    output = keras.layers.Conv2D(64,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    # Conv Layer 3
    output = keras.layers.Conv2D(128,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    #Conv Layer 4
    output = keras.layers.Conv2D(128,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    # Conv Layer 5
    output = keras.layers.Conv2D(256,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    
    output = keras.layers.Flatten()(output)
    #Dense Layer 1
    output = keras.layers.Dense(units=4000,activation="relu")(output)
    #Dense Layer 2
    output = keras.layers.Dense(units=4000,activation="relu")(output)
    #Dense Layer 3
    output = keras.layers.Dense(29,activation="softmax")(output)
    return keras.Model(insert,output)

def generate_Model_no_pool(inputShape,cov_layers,dens_layers,filter_size):
    insert = keras.layers.Conv2D(inputShape,kernel_size=filter_size,activation="relu")(input)
    output=0
    # TODO: layer size Berechnen
    for i in range(0,cov_layers):
        output = keras.layers.Conv2D(0,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.Flatten()
    for i in range(0,dens_layers):
        output = keras.layers.Dense(0,activation="relu")(output)
    return keras.Model(insert,output)

def compile_Model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],

    )
    return model
