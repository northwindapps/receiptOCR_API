#!/bin/bash

folder="C:\Users\ABC\Documents\receiptYOLOProject\cnndata\images"

# Loop through all files in the folder
for file in "$folder"/*; do
    # Get filename only (without path)
    fname=$(basename "$file")

    # Replace all commas with dots
    newname="${fname//,/.}"

    # Only rename if different
    if [[ "$fname" != "$newname" ]]; then
        mv -v "$folder/$fname" "$folder/$newname"
    fi
done
