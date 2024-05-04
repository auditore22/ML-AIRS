import datetime
import os

import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.utils import compute_class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras.callbacks import LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

from Model.Generation.generate_model import build_model
from Model.Training.results.results_insights import results_insights
from Model.Training.results.show_plot_history import plot_training_history, save_confusion_matrix

rescale_factor = 1. / 255


# Learning callback
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def get_training_generator(training_df):
    # Define the data augmentation parameters
    # Adjust these parameters based on your specific dataset and requirements
    data_gen_args = dict(
        rescale=rescale_factor,  # Normalize pixel values
        rotation_range=15,
        width_shift_range=0.1,  # Range (as a fraction of total width) for horizontal shifts
        height_shift_range=0.1,  # Range (as a fraction of total height) for vertical shifts
        shear_range=0.1,  # Simulate a slight slant
        zoom_range=0.15,  # Range for random zoom
        horizontal_flip=True,  # Randomly flip inputs horizontally
        brightness_range=(0.7, 1.2),  # Adjust brightness by 80-120%
        fill_mode='nearest'  # Strategy to fill in newly created pixels
    )

    # Create an ImageDataGenerator object for data augmentation
    image_gen = ImageDataGenerator(**data_gen_args)

    # Create a training generator
    train_generator = image_gen.flow_from_dataframe(
        dataframe=training_df,  # a dataframe containing training data
        x_col='file_path',  # column with image file paths
        y_col='label',  # column with labels
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True,
        drop_remainder=True
    )

    return train_generator


def get_validation_generator(validation_df):
    # Create a separate ImageDataGenerator for validation data (without augmentation)
    validation_image_gen = ImageDataGenerator(rescale=rescale_factor)

    # Similarly, create a validation generator without augmentation
    validation_generator = validation_image_gen.flow_from_dataframe(
        dataframe=validation_df,  # a dataframe containing validation data
        x_col='file_path',
        y_col='label',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False,  # Typically you don't shuffle validation data
        drop_remainder=True
    )

    return validation_generator


def get_callbacks(checkpoint_path):
    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    # Add learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.9,  # Less aggressive reduction
        patience=2,  # Increased patience
        min_lr=1e-7  # Slightly lower minimum learning rate
    )

    # Add a checkpoint callback
    checkpoint = ModelCheckpoint(
        checkpoint_path,  # Unique path for each fold
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        verbose=2
    )

    # Add this callback to your model.fit() call
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=2)

    log_dir = f"Model/Training/logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    return early_stopping, checkpoint, reduce_lr, lr_scheduler, tensorboard


def validate_training(run_name, dataframe, epochs):
    model_dir = f"Model/Training/Models/{run_name}"  # Base directory for saving model files

    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    num_folds = 5
    fold_no = 1
    results = []
    x = dataframe.drop('label', axis=1)  # Features
    y = dataframe['label']  # Labels

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    skf = StratifiedKFold(n_splits=num_folds)
    for train_idx, val_idx in skf.split(x, y):
        train_df = dataframe.iloc[train_idx]
        validation_df = dataframe.iloc[val_idx]

        train_generator = get_training_generator(train_df)
        validation_generator = get_validation_generator(validation_df)

        checkpoint_path = f"{model_dir}/AIR_Model_Fold_{fold_no}.keras"
        early_stopping, checkpoint, reduce_lr, lr_scheduler, tensorboard = get_callbacks(checkpoint_path)

        # Get the best model after tuning
        model = build_model()

        # Start training the model
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, checkpoint, reduce_lr, lr_scheduler, tensorboard],
            class_weight=class_weights_dict,  # Pass the class weights here
            verbose=2
        )

        # Evaluation and prediction
        scores = model.evaluate(validation_generator)
        # Convert scores to numpy if they are tensors
        scores = [score.numpy() if hasattr(score, 'numpy') else score for score in scores]
        results.append(scores)

        y_pred = model.predict(validation_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = validation_generator.classes
        cm = confusion_matrix(y_true, y_pred_classes)

        # Save the confusion matrix and training history
        save_confusion_matrix(cm, fold_no, run_name)
        plot_training_history(history, fold_no, run_name)

        # Generating the classification report
        print(f'Classification Report for Fold {fold_no}:\n{classification_report(y_true, y_pred_classes)}')

        # Optionally save model or log results
        print(
            f"Score for fold {fold_no}: "
            f"{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%")
        fold_no += 1

    results_insights(results)


def train_final_model(run_name, dataframe, epochs, validation_split=0.15):
    model_dir = f"Model/Training/Models/{run_name}"

    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    x = dataframe.drop('label', axis=1)  # Features
    y = dataframe['label']  # Labels
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, stratify=y, random_state=42)

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

    train_generator = get_training_generator(x_train.join(y_train))
    validation_generator = get_validation_generator(x_val.join(y_val))

    checkpoint_path = f"{model_dir}/AIR_Model_Final.keras"
    early_stopping, checkpoint, reduce_lr, lr_scheduler, tensorboard = get_callbacks(checkpoint_path)

    # Get the best model after tuning
    model = build_model()

    # Start training the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint, reduce_lr, lr_scheduler, tensorboard],
        class_weight=class_weights_dict,  # Pass the class weights here
        verbose=2
    )

    # Evaluation and prediction
    scores = model.evaluate(validation_generator)
    # Convert scores to numpy if they are tensors
    scores = [score.numpy() if hasattr(score, 'numpy') else score for score in scores]

    y_pred = model.predict(validation_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes
    cm = confusion_matrix(y_true, y_pred_classes)

    # Save the confusion matrix and training history
    save_confusion_matrix(cm, 0, run_name)
    plot_training_history(history, 0, run_name)

    # Generating the classification report
    print(f'Classification Report for final model:\n{classification_report(y_true, y_pred_classes)}')

    # Log results
    print(
        f"Score for final model: "
        f"{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%")

    model = build_model()  # Rebuild the model architecture
    model.load_weights(checkpoint_path)  # Load the best weights
    save_directory = f"Model/Training/Models/Stable/{run_name}"
    tf.saved_model.save(model, save_directory)

    # Print to confirm saving
    print(f"Model saved successfully at {save_directory}")
