import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.keras"

LEARNING_RATE = 0.0001 # customary learning rate for Adam optimizer
EPOCHS = 40
BATCH_SIZE = 32
NUM_KEYWORDS = 4

def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert to numpy arrays
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y

def get_data_splits(data_path,
                    test_size=0.1, # 10% of data for testing
                    validation_size=0.1): # 10% of data for validation
    X, y = load_dataset(data_path)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # convert inputs from 2d to 3d since CNN expects a 3D array -> (num of segments, num of coefficients (= 13), depth (= 1))
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate=0.0001, error="sparse_categorical_crossentropy"):
    # create the model
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.001))) # relu = rectified linear unit
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same"))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding="same"))

    # flatten the output (3D -> 1D) and feed it into a dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    tf.keras.layers.Dropout(0.3) # shut down 30% of neurons in dense layer to prevent overfitting

    # softmax classifier
    model.add(tf.keras.layers.Dense(NUM_KEYWORDS, activation="softmax")) # i.e. [0.1, 0.7, 0.2] -> [0, 1, 0]

    # compile the model
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model

def main():
    #load train/validation/test data splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build the CNN model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (num of segments, num of coefficients (= 13), depth (= 1))

    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=BATCH_SIZE, epochs=EPOCHS)

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}; Test accuracy: {test_accuracy}")

    # save the model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()