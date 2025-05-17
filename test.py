import Load_data
import Read_Write_Model

def main():
    _,_,test_ds = Load_data.load_data("Data/archive")
    model = Read_Write_Model.Load_model("./models/L7.keras")

    score, acc = model.evaluate(test_ds)

    print("Test score:", score)
    print("Test acc:", acc)

main()
