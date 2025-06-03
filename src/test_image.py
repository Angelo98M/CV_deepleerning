import Read_Write_Model
import keras
import numpy as np

def main():
    shape_template = (200, 200)
    #Load model
    model = Read_Write_Model.Load_model("./models/LT5.keras")
    #Load single image
    img = keras.utils.load_img("./Data/k2.jpg", target_size=shape_template)
    # img = keras.utils.load_img("./Data/archive/test/V_test.jpg", target_size=(200,200))

    #Convert image to numpy array
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.shape)

    #Predition
    pred = model.predict(x)
    #Representation of the predition
    LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nohing", "space"]
    classes = np.argmax(pred, axis=1)
    print("Result: ", LABELS[classes[0]])

main()
