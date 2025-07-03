import CNN_Generator
import Load_data
import Read_Write_Model
import matplotlib.pyplot as plt
from disitillation import Distiller
from disitillation import compile_Distiller

# Train custom model
def custom_model():
    shape_template = (200, 200)
    input_shape = (shape_template[0], shape_template[1], 3)
    #Load data
    train_ds = Load_data.load_train_data("Data/archive2", shape_template)
    val_ds = Load_data.load_validation_data("Data/archive2", shape_template)
    #Create Model
    model = CNN_Generator.create_new_model_v2(input_shape)
    model = CNN_Generator.compile_Model(model)
    #Training
    history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10
            )
    #Plot the results
    plot(history)
    #Save model
    Read_Write_Model.Save_model("./models/vergleich.keras",model)

# Plotting that doesn't work
def plot(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Transfer Learning
def transfer_learning():
    # shape for the images
    shape_template = (200, 200)
    input_shape = (shape_template[0], shape_template[1], 3)

    # Load dataset
    train_ds,val_ds,_ = Load_data.load_data("Data/archive2", shape_template)

    # Create model
    model = CNN_Generator.create_new_model_v2(input_shape)
    model = CNN_Generator.compile_Model(model)

    # Train model
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    # Save model
    Read_Write_Model.Save_model("./models/LT12.keras",model)

    # Model description
    print(model.summary())

#Knowledge distillation
def knowledge_distillation():
    # shape for the images
    shape_template = (200, 200)
    input_shape = (shape_template[0], shape_template[1], 3)

    # Load dataset
    train_ds = Load_data.load_train_data("Data/archive2", shape_template)
    val_ds = Load_data.load_validation_data("Data/archive2", shape_template)

    # Create model
    student = CNN_Generator.generate_Model(input_shape,3,(2,2),2)
    student = CNN_Generator.compile_Model(student)
    teacher = Read_Write_Model.Load_model("./models/LT12.keras")
    teacher.trainable = False
    distiller = Distiller(student=student, teacher=teacher)
    distiller = compile_Distiller(distiller)

    # Train model
    _ = distiller.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
    )

    # Save model
    Read_Write_Model.Save_model("./models/LD39.keras", distiller.student)

    # Model description
    print(distiller.student.summary())


# calls whatever is to be executed
def main():
    # knowledge_distillation()
    custom_model()

if __name__ == "__main__":
    main()
