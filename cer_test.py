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
