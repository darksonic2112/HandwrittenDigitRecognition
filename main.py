import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Define normalization function
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

# Split training set into training and validation sets
train_size = int(0.8 * ds_info.splits['train'].num_examples)
ds_train_and_val = ds_train.shuffle(ds_info.splits['train'].num_examples)

# Split into training and validation sets
ds_train = ds_train_and_val.take(train_size)
ds_val = ds_train_and_val.skip(train_size)

# Apply normalization to images
ds_train = ds_train.map(normalize_img)
ds_val = ds_val.map(normalize_img)

# Batch datasets and prefetch
batch_size = 128
ds_train = ds_train.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# ModelCheckpoint callback to save the best model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# EarlyStopping callback to stop training if no improvement
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1
)

# Train model
history = model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_val,  # Use validation data this time
    callbacks=[checkpoint_callback, early_stopping_callback]
)
print("Training completed.")

# Get training history
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']
epochs = range(1, len(training_loss) + 1)

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, 'b', label='Training loss')
plt.plot(epochs, validation_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, validation_accuracy, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
