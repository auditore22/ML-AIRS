import tensorflow as tf
from keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from kerastuner.tuners import RandomSearch


def tuner_setup():
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,  # Number of different configurations to try
        executions_per_trial=1,  # Number of models to train per trial
        directory='keras_tuner_dir',
        project_name='cat_dog_classification'
    )

    return tuner


def freeze_batch_layers(model):
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def build_model():
    """
    Model that categorizes cat and dog images.
    """
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), pooling='max')
    base_model.trainable = True

    # Add classification head
    model = Sequential([
        base_model,
        BatchNormalization(),
        Dense(64, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.03)),
        Dropout(0.43618),
        Dense(48, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.02)),
        Dropout(0.35597),
        Dense(2, activation='softmax')
    ])

    # Freeze batch normalization layers
    freeze_batch_layers(model)

    # number_of_layers = len(model.layers)
    # print(f"This model has {number_of_layers} layers.")

    # model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
    )
    return model
