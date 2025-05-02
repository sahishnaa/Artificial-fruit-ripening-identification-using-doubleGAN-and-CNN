from google.colab import drive
drive.mount('/content/drive')
import numpy as np

# Define the path to your .npy files
path = '/content/drive/MyDrive/withoutgan/'

# Load the .npy files
class1_images = np.load(path + 'artificialbananas_withoutgan.npy')
class2_images = np.load(path + 'artificialmangoes_withoutgan.npy')
class3_images = np.load(path + 'naturalbanana_withoutgan.npy')
class4_images = np.load(path + 'naturalmangoesresized_withoutgan.npy')
# Create labels for each class
class1_labels = np.zeros((class1_images .shape[0],), dtype=int)
class2_labels = np.ones((class2_images .shape[0],), dtype=int)
class3_labels = np.full((class3_images.shape[0],), 2, dtype=int)
class4_labels = np.full((class4_images.shape[0],), 3, dtype=int)

# Combine the images and labels into single arrays
images = np.concatenate((class1_images, class2_images,class3_images,class4_images), axis=0)
labels = np.concatenate((class1_labels, class2_labels,class3_labels,class4_labels), axis=0)

# Shuffle the data
from sklearn.utils import shuffle
images, labels = shuffle(images, labels, random_state=42)

# Ensure the data is in the correct shape and type
print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

"""
# Load the dataset
# Replace with the actual paths to your .npy files
images = np.load('/path_to_your_npy_files/images.npy')
labels = np.load('/path_to_your_npy_files/labels.npy')

# Check the shape of the dataset
print(f'Images shape: {images.shape}')
print(f'Labels shape: {labels.shape}')"""

# Preprocess the data
# Resize images to 224x224 (common input size for EfficientNet)
images_resized = tf.image.resize(images, [224, 224])

# Normalize images to the range [0, 1]
images_resized = images_resized / 255.0

# Convert labels to categorical (one-hot encoding)
num_classes = 4  # Number of classes in your dataset
labels_categorical = to_categorical(labels, num_classes)

# Convert the TensorFlow tensor to a NumPy array before splitting
images_resized_np = images_resized.numpy() # Convert the tensor to a numpy array

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_resized_np, labels_categorical, test_size=0.2, random_state=42)

# Check the shapes of the split datasets
print(f'Training set shape: {X_train.shape}, {y_train.shape}')
print(f'Validation set shape: {X_val.shape}, {y_val.shape}')

# Build the EfficientNet model using transfer learning
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Create the top model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')  # Assuming 4 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Fit the model with data augmentation
batch_size = 32
epochs = 50

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation accuracy: {val_acc:.4f}')
print(f'Validation loss: {val_loss:.4f}')

# Plot training & validation accuracy and loss
# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
