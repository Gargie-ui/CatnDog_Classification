import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Constants
img_size = 224
batch_size = 32
epochs = 30  # Increased epochs for deeper training

# Callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Load datasets
train_dataset = image_dataset_from_directory(
    'dataset/training_set',
    labels='inferred',
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    'dataset/test_set',
    labels='inferred',
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size
)

class_names = train_dataset.class_names
print("Class indices:", class_names)

# Preprocessing
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(lambda x, y: (preprocess_input(x), y)).prefetch(tf.data.AUTOTUNE)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.2),
])

# Load pretrained base model
base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers

# Build model
inputs = tf.keras.Input(shape=(img_size, img_size, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation='softmax')(x)  # 3 classes: cat, dog, neither

model = Model(inputs, outputs)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs,
    callbacks=[early_stop, lr_reduce]
)

model.save('cat_dog_neither_classifier_new.h5', save_format='h5')
print("✅ Training complete and model saved as .h5.")

