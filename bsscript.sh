#!/bin/bash

folder="C:\Users\ABC\OneDrive\Desktop\receipt_raws"

# Loop through all files in the folder
for file in "$folder"/*; do
    # Get filename only (without path)
    fname=$(basename "$file")

    # Replace all commas with dots
    # newname="${fname//,/.}"
    newname="${fname//_crop_pyttext/}"

    # Only rename if different
    if [[ "$fname" != "$newname" ]]; then
        mv -v "$folder/$fname" "$folder/$newname"
    fi
done
