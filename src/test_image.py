import Read_Write_Model
import keras
import numpy as np
from os import listdir
from os.path import isdir, join

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nohing", "space"]

def test_image(path, model):
    shape_template = (200, 200)
    #Load single image
    img = keras.utils.load_img(path, target_size=shape_template)
    #Convert image to numpy array
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.shape)

    #Predition
    pred = model.predict(x)
    #Representation of the predition
    classes = np.argmax(pred, axis=1)
    print("(", path, ")\nResult: ", LABELS[classes[0]])

def main():
    #Load model
    model = Read_Write_Model.Load_model("./models/LT12.keras")
    for file in listdir("./Data/"):
        if isdir(join("./Data/", file)): continue
        test_image(join("./Data/", file), model)

    

main()
