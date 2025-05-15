import CNN_Generator
import Load_data
import Read_Write_Model


def main():
    (train_ds,val_ds,test_ds) = Load_data.load_data("Data/archive")
    

main()