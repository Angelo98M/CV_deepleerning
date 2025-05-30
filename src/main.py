import CNN_Generator
import Load_data
import Read_Write_Model
import matplotlib.pyplot as plt



def main():
#     #Load data
#     (train_ds,val_ds,test_ds) = Load_data.load_data("Data/archive2")
#     #Create Model
#     model = CNN_Generator.generate_Model((400,400,3),3,(2,2),2)
#     model = CNN_Generator.compile_Model(model)
#     #Training
#     history = model.fit(
#         train_ds,
#         validation_data=val_ds,
#         epochs=30
#     )
#     #Plot the results
#     plot(history)
#     #Save model
#     Read_Write_Model.Save_model("./models/L16.keras",model)

# # Plotting that doesn't work
# def plot(history):
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()

    shape_template = (60, 60)
    input_shape = (shape_template[0], shape_template[1], 3)
    (train_ds,val_ds,test_ds) = Load_data.load_data("Data/archive2", shape_template)
    model = CNN_Generator.create_new_model(input_shape)
    model = CNN_Generator.compile_Model(model)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )
    Read_Write_Model.Save_model("./models/LT2.keras",model)
    print(model.summary())

main()
