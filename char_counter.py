import json
from collections import Counter

json_file=r"C:\Users\ABC\Documents\receiptYOLOProject\cnndata\labels.json"
json_file=r"C:\Users\ABC\Documents\clean_unique\cnndata\labels.json"
# Load your labels.json
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

char_counter = Counter()

for item in data:
    annotation = item["annotation"]
    for char in annotation:
        char_counter[char] += 1

# Print counts
for char, count in sorted(char_counter.items()):
    print(f"'{char}': {count} images")
