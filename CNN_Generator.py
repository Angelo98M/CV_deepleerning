import keras


def generate_Model(inputShape,filter_size,pool_size,pool_stride):
    insert = keras.layers.Input(shape=(inputShape[0],inputShape[1],inputShape[2]))
    output=0
    # Conv Layer 1
    output = keras.layers.Conv2D(16,kernel_size=filter_size,activation="relu")(insert)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    #Conv Layer 2
    output = keras.layers.Conv2D(32,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    # Conv Layer 3
    output = keras.layers.Conv2D(64,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    #Conv Layer 4
    output = keras.layers.Conv2D(128,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    # Conv Layer 5
    output = keras.layers.Conv2D(256,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    
    output = keras.layers.Flatten()(output)
    #Dense Layer 1
    output = keras.layers.Dense(units=256,activation="relu")(output)
    #Dense Layer 2
    output = keras.layers.Dense(units=128,activation="relu")(output)
    #Dense Layer 3
    output = keras.layers.Dense(units=64,activation="relu")(output)
    #Dense Layer 4
    output = keras.layers.Dense(29,activation="softmax")(output)
    return keras.Model(insert,output)

#lazy wrapper for compiling
def compile_Model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],

    )
    return model
