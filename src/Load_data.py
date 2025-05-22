import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data=[]
train=[]
val=[]

def load_data(path):
    train_ds = keras.utils.image_dataset_from_directory( 
        directory= path+"/train",
        labels= "inferred",
        label_mode = "int",
        class_names= None,
        color_mode="rgb",
        batch_size=32,
        image_size=(400, 400),
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
        image_size=(400, 400),
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
        image_size=(400, 224),
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
    
