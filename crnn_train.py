import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os

img_height, img_width = 62, 292

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # load grayscale
    img = img.astype(np.float32) / 255.0          # normalize 0–1
    img = np.expand_dims(img, axis=-1)            # add channel dimension
    return img

#Prepare Images
image_paths = ["img1.png", "img2.png", "img3.png"]
batch_images = np.array([load_image(p) for p in image_paths])  # shape: (batch, H, W, 1)


# -------------------
# Config
# -------------------
img_height = 32
img_width = 128
num_channels = 1
num_classes = 38  # 26 letters + 10 digits + blank + etc.
max_label_length = 5  # maximum text length for dummy data

# -------------------
# Build CRNN backbone
# -------------------
inputs = layers.Input(name="input", shape=(img_height, img_width, num_channels), dtype="float32")

x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)  # vertical pool only

x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,1))(x)

# CNN output shape: (batch, h, w, c)
# reshape → (batch, time_steps=w, features=h*c)
new_shape = (-1, x.shape[1] * x.shape[3])
x = layers.Reshape(target_shape=new_shape)(x)

# RNN part
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

base_model = models.Model(inputs, outputs, name="CRNN")

# -------------------
# Add CTC loss layer
# -------------------
labels = layers.Input(name="label", shape=(max_label_length,), dtype="int64")
input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name="ctc")(
    [outputs, labels, input_length, label_length]
)

training_model = models.Model(inputs=[inputs, labels, input_length, label_length],
                              outputs=loss_out)

training_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

print("✅ Models built: base_model (for inference), training_model (with CTC loss)")

# -------------------
# Dummy Data
# -------------------
num_samples = 100
X_dummy = np.random.rand(num_samples, img_height, img_width, num_channels).astype(np.float32)

# random integer labels (simulate text as numbers between 1 and num_classes-1)
# valid characters: 1 .. num_classes-2  (leave num_classes-1 for CTC blank)
y_dummy = np.random.randint(1, num_classes - 1, size=(num_samples, max_label_length))

# lengths
input_lengths = np.ones((num_samples, 1)) * (outputs.shape[1])  # timesteps from CNN
label_lengths = np.random.randint(1, max_label_length+1, size=(num_samples, 1))

# -------------------
# Train on dummy data
# -------------------
training_model.fit(
    x={"input": X_dummy,
       "label": y_dummy,
       "input_length": input_lengths,
       "label_length": label_lengths},
    y=np.zeros(num_samples),  # dummy because Lambda gives loss
    batch_size=8,
    epochs=2
)

# -------------------
# Inference / Decoding
# -------------------
preds = base_model.predict(X_dummy[:2])  # predict first 2 samples
decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1])

print("🔤 Decoded predictions:")
print(decoded[0].numpy())
