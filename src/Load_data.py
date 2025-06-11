import keras

def load_data(path, shape=(224, 224)):
    train_ds = keras.utils.image_dataset_from_directory( 
        directory= path+"/train",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=shape,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        # interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,

    )
    val_ds = keras.utils.image_dataset_from_directory( 
        directory= path+"/validation",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=shape,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        # interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,

    )
    test_ds = keras.utils.image_dataset_from_directory( 
        directory= path+"/test",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=shape,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        # interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,

    )

    return train_ds, val_ds,test_ds
    
def load_train_data(path, shape=(224, 224)):
    dataset = keras.utils.image_dataset_from_directory( 
        directory= path+"/train",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=shape,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        # interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,
    )
    return dataset

def load_validation_data(path, shape=(224, 224)):
    dataset = keras.utils.image_dataset_from_directory( 
        directory= path+"/validation",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=shape,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        # interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,
    )
    return dataset

def load_test_data(path, shape=(224, 224)):
    dataset = keras.utils.image_dataset_from_directory( 
        directory= path+"/test",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=shape,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        # interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
        data_format=None,
        verbose=True,
    )
    return dataset
