import keras

# Load trainings-, Valadation- and Test datasets form given Path
def load_data(path, shape=(224, 224)):
    """
    Loads dataset from path

    Parameters:
        path: string
            path to the dataset (requires subdirectories 'train', 'test' and 'validation')
        shape: tuple
            tuple with two items for the image dimensions

    Returns:
        tuple containing the training, validation and test datasets
    """

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
   
#Load Only Trainigns Data form given path 
def load_train_data(path, shape=(224, 224)):
    """
    Loads dataset from path

    Parameters:
        path: string
            path to the dataset (requires subdirectories 'train', 'test' and 'validation')
        shape: tuple
            tuple with two items for the image dimensions

    Returns:
        training dataset
    """

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

#Load Only Validation Data form given path 
def load_validation_data(path, shape=(224, 224)):
    """
    Loads dataset from path

    Parameters:
        path: string
            path to the dataset (requires subdirectories 'train', 'test' and 'validation')
        shape: tuple
            tuple with two items for the image dimensions

    Returns:
        validation dataset
    """

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

#Load Only Test Data form given path 
def load_test_data(path, shape=(224, 224)):
    """
    Loads dataset from path

    Parameters:
        path: string
            path to the dataset (requires subdirectories 'train', 'test' and 'validation')
        shape: tuple
            tuple with two items for the image dimensions

    Returns:
        test dataset
    """

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
