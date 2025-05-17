import CNN_Generator
import Load_data
import Read_Write_Model
import matplotlib.pyplot as plt



def main():
    (train_ds,val_ds,test_ds) = Load_data.load_data("Data/archive")
    model = CNN_Generator.generate_Model((200,200,3),3,(2,2),2)
    model = CNN_Generator.compile_Model(model)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30
    )
    plot(history)
    Read_Write_Model.Save_model("./models/L8.keras",model)

def plot(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

main()
