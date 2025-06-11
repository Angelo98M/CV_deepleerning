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
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"],

    )
    return model

def create_new_model(image_shape):
    rest = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape = image_shape,
        name="restnet50"
    )
    for layer in rest.layers:
        layer.trainable = False
        
    
    model = keras.Sequential()
    model.add(rest)
    #model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=512,activation="relu"))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Dense(units=4096,activation="relu"))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Dense(units=29,activation="softmax"))

    return model

def create_new_model_v2(image_shape):
    rest = keras.applications.ResNet101(
        include_top=False,
        weights="imagenet",
        input_shape = image_shape,
        name="restnet101"
    )
    for layer in rest.layers:
        layer.trainable = False


    model = keras.Sequential()
    model.add(keras.layers.RandomFlip("horizontal_and_vertical"))
    model.add(keras.layers.RandomRotation(0.4))
    model.add(keras.layers.RandomZoom(0.4))
    model.add(keras.layers.RandomContrast(0.4))
    model.add(keras.layers.RandomBrightness(0.4))
    model.add(keras.layers.RandomTranslation(0.4,0.3))
    model = keras.Sequential()
    model.add(rest)
    #model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=512,activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=1024,activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=2048,activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(units=29,activation="softmax"))

    return model
