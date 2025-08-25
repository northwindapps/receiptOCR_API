cd "/c/Users/ABC/OneDrive/Desktop/cnndata/images"
# C:\Users\ABC\OneDrive\Desktop\receipt_augments
idx=1
for f in *.jpg; do
    # Remove capital letters (optional)
    new=$(echo "$f" | tr -d 'A-Za-z' | tr -d '[:space:]')

    # Add index before .jpg
    base="${new%.jpg}"
    # reduce underscores
    base=$(echo "$base" | sed 's/_\+/_/g')
    newname="${base}_${idx}.jpg"

    # Rename
    mv "$f" "$newname"

    ((idx++))
done
