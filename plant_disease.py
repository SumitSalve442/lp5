# plant_disease_detection_cnn.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# 1. Dataset Paths
train_path = "Plant_Disease_Dataset/train"
valid_path = "Plant_Disease_Dataset/valid"
test_path  = "Plant_Disease_Dataset/test"


# 2. Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


# 3. Load Dataset
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    valid_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)



# 4. Build CNN Model
model = Sequential()

# Convolution Layer 1
model.add(Conv2D(
    32,
    (3, 3),
    activation='relu',
    input_shape=(128, 128, 3)
))
model.add(MaxPooling2D(2, 2))

# Convolution Layer 2
model.add(Conv2D(
    64,
    (3, 3),
    activation='relu'
))
model.add(MaxPooling2D(2, 2))

# Convolution Layer 3
model.add(Conv2D(
    128,
    (3, 3),
    activation='relu'
))
model.add(MaxPooling2D(2, 2))

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(
    128,
    activation='relu'
))

# Dropout Layer
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(
    train_data.num_classes,
    activation='softmax'
))


# 5. Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# 6. Model Summary
model.summary()


# 7. Train Model
history = model.fit(
    train_data,
    epochs=3,
    validation_data=valid_data
)


# 8. Evaluate Model
test_loss, test_accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", test_accuracy)


# 9. Accuracy Graph
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("CNN Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()