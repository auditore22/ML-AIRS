import os

import matplotlib.pyplot as plt
from seaborn import heatmap


def save_confusion_matrix(cm, fold_no, run_name, base_output_dir="Model/Training/Plots"):
    output_dir = f"{base_output_dir}/{run_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.figure(figsize=(10, 7))
    heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    if fold_no == 0:
        plt.title(f'Confusion Matrix for Run {run_name}')
        plt.savefig(f"{output_dir}/Confusion_Matrix_{run_name}.png")
    else:
        plt.title(f'Confusion Matrix for Fold {fold_no}')
        plt.savefig(f"{output_dir}/Confusion_Matrix_Fold_{fold_no}.png")
    plt.close()


def plot_training_history(history, fold_no, run_name, base_output_dir="Model/Training/Plots"):
    output_dir = f"{base_output_dir}/{run_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    # Plot training & validation loss values
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    if fold_no == 0:
        plt.savefig(f"{output_dir}/Training_History_{run_name}.png")
    else:
        plt.savefig(f"{output_dir}/Training_History_Fold_{fold_no}.png")
    plt.close()
