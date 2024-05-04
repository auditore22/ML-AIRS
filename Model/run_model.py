import pandas as pd


from Model.Training.train_model import validate_training, train_final_model




def validate_dataframe(dataframe):
    if dataframe.empty:
        raise ValueError("The DataFrame is empty.")
    if 'label' not in dataframe.columns:
        raise ValueError("The 'label' column is missing from the DataFrame.")
    if dataframe['label'].isna().any():
        raise ValueError("The 'label' column contains missing values.")
    unique_labels = dataframe['label'].unique()
    if len(unique_labels) < 2:
        raise ValueError("There should be at least two unique labels for classification.")


def run():
    # Specify the path to your CSV file
    csv_file_path = 'ImageHandling/image_labels.csv'

    # Load the DataFrame
    df = pd.read_csv(csv_file_path)
    validate_dataframe(df)

    validate_training("run1", df, 50)  # Find the best parameters

    # train_final_model('run_final', df, 100)  # example parameters # Generate best model with previous parameters

