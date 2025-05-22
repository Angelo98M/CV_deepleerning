import Load_data
import Read_Write_Model

def main():
    #Load data
    _,_,test_ds = Load_data.load_data("Data/archive2")
    #Load model
    model = Read_Write_Model.Load_model("./models/L14.keras")

    #Evaluate the model
    score, acc = model.evaluate(test_ds)

    print("Test score:", score)
    print("Test acc:", acc)

main()
