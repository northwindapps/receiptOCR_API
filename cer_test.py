vocab = "0123456789.,-$;/￥ " # Added space to vocab

# -------------------
# Vocabulary mapping
# -------------------
char_to_idx = {c:i+1 for i,c in enumerate(vocab)}  # 0 reserved for blank
idx_to_char = {i+1:c for i,c in enumerate(vocab)}

# 1) simple edit distance
def edit_distance(a, b):
    # Levenshtein
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb+1):
            cur = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j-1], dp[j])
            prev = cur
    return dp[lb]

# 2) decode helper (use your base_model)
def decode_batch(y_pred):
    # y_pred: (B, T, num_classes)
    input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
    decoded = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)[0][0]
    decoded = tf.keras.backend.get_value(decoded)
    out = []
    for seq in decoded:
        txt = ''.join([idx_to_char.get(int(i), '') for i in seq if int(i) > 0])
        out.append(txt)
    return out

# 1) Make sure you have your test image paths and ground-truth texts
test_paths = test_img  # list of test image file paths
test_texts = test_labels  # list of corresponding strings

# 2) Sample CER calculation
chars_total = 0
dist_total = 0

for img_path, gt_text in zip(test_paths, test_texts):
    # Load image
    im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ratio = img_height / im.shape[0]
    new_w = int(im.shape[1] * ratio)
    im = cv2.resize(im, (new_w, img_height)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=-1)  # add channel
    im = np.expand_dims(im, axis=0)   # add batch dimension

    # Predict
    y_pred = base_model.predict(im)
    pred_text = decode_batch(y_pred)[0]

    # Compute edit distance
    dist_total += edit_distance(gt_text, pred_text)
    chars_total += len(gt_text)

# Compute CER
cer_test = dist_total / max(1, chars_total)
print(f"Test CER: {cer_test:.3f} ({dist_total}/{chars_total})")
