import Read_Write_Model
import keras
import numpy as np
from os import listdir
from os.path import isdir, join

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"]

def test_image_identification(path, model, label):
    shape_template = (200, 200)
    #Load single image
    img = keras.utils.load_img(path, target_size=shape_template)
    #Convert image to numpy array
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #Predition
    pred = model.predict(x, verbose=0)
    #Representation of the predition
    classes = np.argmax(pred, axis=1)
    if label == LABELS[classes[0]]: return
    print("(", path, ")\nResult: ", LABELS[classes[0]])

def main():
    model = Read_Write_Model.Load_model("./models/LT12.keras")

    for dir in listdir("./Data/archive2/test/"):
        dir_path = join("./Data/archive2/test/", dir)
        if not isdir(dir_path): continue
        for file in listdir(dir_path):
            file_path = join(dir_path, file)
            if isdir(file_path): continue
            test_image_identification(file_path, model, dir)

if __name__ == "__main__":
    main()
