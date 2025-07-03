import Load_data
import Read_Write_Model

def main():
    #Load data
    test_ds = Load_data.load_test_data("Data/archive3", (200, 200))
    #Load model
    model = Read_Write_Model.Load_model("./models/LD38.keras")

    #Evaluate the model
    score, acc = model.evaluate(test_ds)

    print("Test score:", score)
    print("Test acc:", acc)

main()
