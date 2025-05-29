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

    model = CNN_Generator.create_new_model((200,200,3))
    print(model.summary())

main()
