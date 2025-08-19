import tensorflow as tf
from tensorflow.keras import layers, models

# Input image shape: (height, width, channels)
input_shape = (32, 128, 1)  # grayscale 32x128

num_classes = 38  # 26 letters + 10 digits + CTC blank + special symbols as needed

inputs = layers.Input(name="input", shape=input_shape, dtype="float32")

# --- CNN feature extractor ---
x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,2))(x)

x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)  # pool only vertically

x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,1))(x)

# --- Reshape for RNN ---
# Combine height and channels → make it (batch, width, features)
x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)


# --- RNN layers ---
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

# --- Dense output ---
x = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=x, name="CRNN")
model.summary()

# --- CTC Loss ---
labels = layers.Input(name="label", shape=(None,), dtype="float32")
input_length = layers.Input(name="input_length", shape=(1,), dtype="int64")
label_length = layers.Input(name="label_length", shape=(1,), dtype="int64")

def ctc_lambda(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name="ctc")(
    [x, labels, input_length, label_length]
)

training_model = models.Model(
    inputs=[inputs, labels, input_length, label_length],
    outputs=loss_out
)

training_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

print("✅ CRNN model ready")
