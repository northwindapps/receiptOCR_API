import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------
# Config
# -------------------
img_height, img_width = 62, 292
num_channels = 1
vocab = "0123456789.,$"
num_classes = len(vocab) + 2  # +1 for padding, +1 for CTC blank
max_label_len = 10
batch_size = 8

# -------------------
# Vocabulary mapping
# -------------------
char_to_idx = {c:i+1 for i,c in enumerate(vocab)}  # 0 reserved for blank
idx_to_char = {i+1:c for i,c in enumerate(vocab)}

def text_to_labels(text):
    return [char_to_idx[c] for c in text]

# -------------------
# Data loading
# -------------------
data_path = r"C:\Users\ABC\Documents\receiptYOLOProject\cnndata\images"
label_file = r"C:\Users\ABC\Documents\receiptYOLOProject\cnndata\labels.txt"

image_paths, label_sequences = [], []
with open(label_file, "r") as f:
    for line in f:
        fname, text = line.strip().split(",")
        image_paths.append(os.path.join(data_path, fname))
        label_sequences.append(list(text_to_labels(text)))

# -------------------
# Data generator
# -------------------
def data_generator(batch_size):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_labels = label_sequences[i:i+batch_size]

            X, Y, label_len = [], [], []
            for img_path, lbl in zip(batch_paths, batch_labels):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                X.append(img)
                Y.append(lbl)
                label_len.append(len(lbl))

            X = np.array(X, dtype=np.float32)
            Y_padded = np.zeros((len(Y), max_label_len), dtype=np.int32)
            for j, lbl in enumerate(Y):
                Y_padded[j, :len(lbl)] = lbl

            input_len = np.ones((len(X),1)) * (img_width // 4)
            label_len_arr = np.array(label_len).reshape(-1,1)

            yield {"input": X,
                   "label": Y_padded,
                   "input_length": input_len,
                   "label_length": label_len_arr}, np.zeros(len(X))

# -------------------
# Positional embedding layer
# -------------------
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        T = tf.shape(x)[1]
        positions = tf.range(start=0, limit=T, delta=1)
        pe = self.pos_emb(positions)
        pe = tf.expand_dims(pe, axis=0)
        return x + pe

# -------------------
# Transformer block
# -------------------
def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# -------------------
# Build CNN → Transformer → CTC
# -------------------
inputs = layers.Input(shape=(img_height, img_width, num_channels), name="input")

# CNN backbone
x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)
x = layers.Conv2D(512, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,1))(x)

# Reshape CNN output to (B, T, D)
H_prime, W_prime, C_prime = x.shape[1], x.shape[2], x.shape[3]
x = layers.Permute((2,1,3))(x)  # (B, W', H', C')
x = layers.Reshape((W_prime, H_prime*C_prime))(x)  # (B, T, D)

# Linear projection + positional embedding
model_dim = 256
x = layers.Dense(model_dim)(x)
x = PositionalEmbedding(max_len=W_prime, d_model=model_dim)(x)

# Transformer encoder stack
x = transformer_block(x, head_size=64, num_heads=4, ff_dim=256)
x = transformer_block(x, head_size=64, num_heads=4, ff_dim=256)

# Final classification per timestep
outputs = layers.Dense(num_classes, activation="softmax")(x)

base_model = models.Model(inputs, outputs, name="TransformerOCR")

# -------------------
# CTC Loss
# -------------------
labels = layers.Input(name="label", shape=(max_label_len,), dtype="int32")
input_length = layers.Input(name="input_length", shape=(1,), dtype="int32")
label_length = layers.Input(name="label_length", shape=(1,), dtype="int32")

def ctc_lambda(args):
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

loss_out = layers.Lambda(ctc_lambda, output_shape=(1,), name="ctc")(
    [outputs, labels, input_length, label_length]
)

training_model = models.Model(
    inputs=[inputs, labels, input_length, label_length],
    outputs=loss_out
)

training_model.compile(optimizer="adam", loss={"ctc": lambda y_true, y_pred: y_pred})

# -------------------
# Training
# -------------------
steps_per_epoch = len(image_paths) // batch_size
training_model.fit(
    data_generator(batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=10
)

# -------------------
# Inference example
# -------------------
img = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
img = np.expand_dims(img, axis=(0,-1))
preds = base_model.predict(img)
decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0])*preds.shape[1])
print("Decoded:", [idx_to_char[i] for i in decoded[0].numpy()[0] if i!=0])
