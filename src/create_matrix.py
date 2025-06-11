import Load_data
import Read_Write_Model
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nohing", "space"]

def create_confusion_matrix(model, dataset):
    y_pred_labels = np.array([])
    y_true = np.array([])
    for x, y in dataset:
        y_pred_labels = np.concatenate([y_pred_labels,
                                        np.argmax(model.predict(x, verbose=0), axis=-1)])
        y_true = np.concatenate([y_true, y.numpy()])

    return confusion_matrix(y_true, y_pred_labels)

def main():
    shape_template = (200, 200)
    model = Read_Write_Model.Load_model("./models/LT10.keras")
    dataset = Load_data.load_test_data("Data/archive2", shape_template)
    cm = create_confusion_matrix(model, dataset)
    sns.heatmap(cm, annot=True, fmt='d', yticklabels=LABELS, xticklabels=LABELS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()
