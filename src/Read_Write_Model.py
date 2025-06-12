import keras

# Load Model form given Path
def Load_model(path):
    return keras.models.load_model(path)

#Load Weigths form given Path
def Load_weigths(path,model):
    model.load_weights(path)
    return model

# Save weigths from Model into given Path
def Save_weigths(path,model):
    model.save_weigths(path)

# Save given Model at given Path
def Save_model(path,model):
    model.save(path)