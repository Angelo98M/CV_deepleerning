import keras


def generate_Model(inputShape,filter_size,pool_size,pool_stride):
    insert = keras.layers.Conv2D(inputShape,kernel_size=filter_size,activation="relu")(input)
    output=0
    # TODO: layer size Berechnen
    # Conv Layer 1
    output = keras.layers.Conv2D(8,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    #Conv Layer 2
    output = keras.layers.Conv2D(12,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    # Conv Layer 3
    output = keras.layers.Conv2D(20,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    #Conv Layer 4
    output = keras.layers.Conv2D(28,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    # Conv Layer 5
    output = keras.layers.Conv2D(40,kernel_size=filter_size,activation="relu")(output)
    output = keras.layers.MaxPool2D(pool_size=pool_size,strides=pool_stride)(output)
    
    output = keras.layers.Flatten()
    #Dense Layer 1
    output = keras.layers.Dense(inputShape[1]*inputShape[0]*40,activation="relu")(output)
    #Dense Layer 2
    output = keras.layers.Dense(29,activation="relu")(output)
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