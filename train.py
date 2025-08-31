import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------
# Data Augmentation for Training
# -------------------------------
trainDataGen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode="nearest"
)

# Only Rescaling for Validation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training Data Generator
trainGenerator = trainDataGen.flow_from_directory(
    "datasets/Train",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# Validation Data Generator
validation_generator = test_datagen.flow_from_directory(
    "datasets/Test",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# -------------------------------
# Build CNN Model
# -------------------------------
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# Layer 2
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# Layer 3
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# Layer 4
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(46, activation="softmax"))  # 46 Devanagari classes

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

# -------------------------------
# Train Model
# -------------------------------
res = model.fit(
    trainGenerator,
    epochs=25,
    validation_data=validation_generator
)

# -------------------------------
# Save Model
# -------------------------------
model.save("HindiModel.h5")
print("✅ Model saved as HindiModel.h5")
