import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


"""
The first lines are for initializing the data that is trained and tested on.
We have two tuples of train and test data, that loads in the mnist dataset from tensorflow.
This dataset contains labeled data with 10 different classes (the decimal values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

The images are converted to floats and are divided by 255, in order to normalize the values to [0, 1].
After that we expand these to tensors.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

"""
The following lines represent the model structure.
It consists of 2 convolutional layers, with 32 and 64 filters respectively, each supporting a 3x3 kernel.
The activation function is relu, while the input_shape is set to 28, 28, 1, representing the width, height and number of channels.

At the end a fully connected (dense) layer is used for the classification.
The dropout is used for further exploration of different neurons.
"""
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

"""
The next lines set up a directory for storing model checkpoints, construct the filepath for saving checkpoints, 
and initialize a ModelCheckpoint callback to save the best model based on validation accuracy during training.

Make sure the filepath ends with .keras or you get an error :))))))))))))))))
"""
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_filepath = os.path.join(checkpoint_dir, 'best_model.keras')

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

"""
In the next lines the actual training is happening.

The amount of epochs specifies the number of iterations over the entire dataset for which the model will be trained.
This validation_split specifies the fraction of the training data to be used as validation data. 
In this case, 20% of the training data will be used for validation during training.

Lastly, after the training is complete, the model is evaluated on the test data, and the loss and accuracy of the model is determined.
"""
history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[model_checkpoint_callback])

model = tf.keras.models.load_model(checkpoint_filepath)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

"""
At the end a plot is created to visualize the progress between each epochs.
Specifically the metrics Model loss and model accuracy during the training will be highlighted.
"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
