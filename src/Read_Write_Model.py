import keras

def Load_model(path):
    return keras.models.load_model(path)

def Load_weigths(path,model):
    model.load_weights(path)
    return model

def Save_weigths(path,model):
    model.save_weigths(path)

def Save_model(path,model):
    model.save(path)