# CRNN + Transformer (CTC) — runnable example
# -------------------------------------------
# Works with TensorFlow 2.10+ (MultiHeadAttention available).
# Uses dummy data; swap X/y with your real OCR dataset.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------
# Config
# -------------------
img_height = 32
img_width  = 128
num_channels = 1
num_classes  = 38   # 26 letters + 10 digits + (ctc blank, etc)
max_label_length = 5

# -------------------
# Positional Embedding (learnable)
# -------------------
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        # x: (B, T, D)
        T = tf.shape(x)[1]
        positions = tf.range(start=0, limit=T, delta=1)
        pe = self.pos_emb(positions)  # (T, D)
        pe = tf.expand_dims(pe, axis=0)  # (1, T, D)
        return x + pe

# -------------------
# Transformer Encoder Block
# -------------------
def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    # Self-attention
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-forward
    ffn = layers.Dense(ff_dim, activation="relu")(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    return x

# -------------------
# Build CNN → Sequence
# -------------------
inputs = layers.Input(name="input", shape=(img_height, img_width, num_channels), dtype="float32")

x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D((2,2))(x)  # -> (16, 64)

x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,2))(x)  # -> (8, 32)

x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
x = layers.MaxPooling2D((2,1))(x)  # -> (4, 32)  (vertical down, keep width)

x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,1))(x)  # -> (2, 32)

# Now x: (B, H', W', C') = (B, 2, 32, 512)
# We want sequence along width → time steps = W' (= 32 here)
# Reshape to (B, T, D) where T=W', D=H'*C' (= 2*512 = 1024)
H_prime = x.shape[1]        # 2
W_prime = x.shape[2]        # 32  (this becomes the time_steps for CTC)
C_prime = x.shape[3]        # 512
x = layers.Permute((2,1,3))(x)                     # (B, W', H', C')
x = layers.Reshape((W_prime, H_prime*C_prime))(x)  # (B, T=W', D=H'*C')

# Optional linear projection to a smaller model width for the Transformer
model_dim = 256
x = layers.Dense(model_dim)(x)  # (B, T, 256)

# Add positional information
x = PositionalEmbedding(max_len=img_width // 4, d_model=model_dim)(x)  # max_len just needs to cover T

# -------------------
# Transformer Encoder stack (replace/augment LSTMs)
# -------------------
# If you want to keep BiLSTMs too, you can add them before the transformer:
# x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)

# Pure Transformer encoder (2 blocks as example)
x = transformer_block(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)
x = transformer_block(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1)

# Final classification per time step
outputs = layers.Dense(num_classes, activation="softmax")(x)  # (B, T, num_classes)

base_model = models.Model(inputs, outputs, name="CRNN_Transformer")

# -------------------
# CTC Loss plumbing
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

training_model = models.Model(
    inputs=[inputs, labels, input_length, label_length],
    outputs=loss_out,
    name="CRNN_Transformer_Training"
)

training_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                       loss={"ctc": lambda y_true, y_pred: y_pred})

print("✅ Models built: base_model (inference), training_model (CTC-loss)")

# -------------------
# Dummy Data (for sanity check)
# -------------------
num_samples = 100
X_dummy = np.random.rand(num_samples, img_height, img_width, num_channels).astype(np.float32)

# labels in [1, num_classes-1); reserve (num_classes-1) as CTC blank implicitly
y_dummy = np.random.randint(1, num_classes - 1, size=(num_samples, max_label_length))

# CTC time steps come from base_model output sequence length:
time_steps = base_model.output_shape[1]  # should be 32 given the pooling above
input_lengths = np.full((num_samples, 1), fill_value=time_steps, dtype=np.int64)
label_lengths = np.random.randint(1, max_label_length+1, size=(num_samples, 1), dtype=np.int64)

# -------------------
# Train on dummy data
# -------------------
training_model.fit(
    x={
        "input": X_dummy,
        "label": y_dummy,
        "input_length": input_lengths,
        "label_length": label_lengths
    },
    y=np.zeros((num_samples,)),  # Dummy; the Lambda already returns the loss
    batch_size=8,
    epochs=2,
    verbose=1
)

# -------------------
# Inference / Decoding (greedy CTC)
# -------------------
preds = base_model.predict(X_dummy[:2])
decoded, _ = tf.keras.backend.ctc_decode(preds, input_length=np.ones(preds.shape[0]) * preds.shape[1])
print("🔤 Decoded predictions (greedy indices):")
print(decoded[0].numpy())
