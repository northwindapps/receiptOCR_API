import numpy as np

def inspect_model_output(output):
    try:
        print('Detailed output inspection:')

        batch_output = output[0]  # remove batch dimension
        num_rows, num_cols = batch_output.shape

        # Check for any non-zero values and find max value + position
        has_non_zero_values = np.any(batch_output != 0)
        max_value = np.max(batch_output)
        max_pos = np.unravel_index(np.argmax(batch_output), batch_output.shape)

        print(f'  Has non-zero values: {has_non_zero_values}')
        print(f'  Max value: {max_value:.6f} at position {max_pos}')

        # Print values around max value (up to 2 rows/cols before/after)
        start_row = max(0, max_pos[0] - 2)
        end_row = min(num_rows - 1, max_pos[0] + 2)
        start_col = max(0, max_pos[1] - 2)
        end_col = min(num_cols - 1, max_pos[1] + 2)

        print('  Values around maximum:')
        for r in range(start_row, end_row + 1):
            row_vals = batch_output[r, start_col:end_col + 1]
            row_str = ' '.join(f'{v:.4f}' for v in row_vals)
            print(f'    Row {r}: {row_str}')

        # Print first 5 potential detections (columns represent predictions)
        print('  First 5 potential detections:')
        for i in range(min(5, num_cols)):
            cx, cy, w, h = batch_output[0:4, i]

            # Find max class probability from class probs starting at row 4
            class_probs = batch_output[4:, i]
            max_class_prob = np.max(class_probs)
            max_class_index = np.argmax(class_probs)

            print(
                f'    Detection {i}: box=[{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}], '
                f'class={max_class_index}, prob={max_class_prob:.4f}'
            )

        # Count detections with confidence > 0.1 (debugging threshold)
        high_conf_count = 0
        for i in range(num_cols):
            class_probs = batch_output[4:, i]
            if np.max(class_probs) > 0.1:
                high_conf_count += 1
        print(f'  Detections with confidence > 0.1: {high_conf_count}')

    except Exception as e:
        print(f'Error in output inspection: {e}')


def check_high_confidence_detection(output, index=8194):
    try:
        batch_output = output[0]
        cx = batch_output[0, index]
        cy = batch_output[1, index]
        w = batch_output[2, index]
        h = batch_output[3, index]

        print(f'High confidence detection at index {index}:')
        print(f'  Box coordinates: x={cx}, y={cy}, w={w}, h={h}')

        class_probs = batch_output[4:, index]
        print(f'  First 10 class probabilities: {class_probs[:10]}')

        max_class_index = np.argmax(class_probs)
        max_class_prob = class_probs[max_class_index]
        print(f'  Max class: {max_class_index}, probability: {max_class_prob}')
    except Exception as e:
        print(f'Error checking high confidence detection: {e}')
