#!/bin/bash

folder="/c/Users/ABC/Documents/receiptYOLOProject/cnndata/images"
text_file="/c/Users/ABC/Documents/receiptYOLOProject/cnndata/labels.txt"

# Empty the output file
> "$text_file"

# Loop through all files in the folder
for fname in "$folder"/*; do
    base=$(basename "$fname")   # get filename only

    # Skip if filename contains "labels"
    if [[ "$base" == *labels* ]]; then
        continue
    fi

    # Split by underscore
    IFS='_' read -ra parts <<< "$base"
    annotation="${parts[1]}"   # second chunk

    # Write to text file
    echo "$base,$annotation" >> "$text_file"
done
