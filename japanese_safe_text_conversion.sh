#!/bin/bash
shopt -s nullglob

# Change to your target directory
cd "/c/Users\ABC\Documents\clean_unique_kanji\clean" || exit

for f in *年*月*日*; do
    newname="${f//年/yy}"
    newname="${newname//月/mm}"
    newname="${newname//日/dd}"
    mv -- "$f" "$newname"
    echo "Renamed: $f → $newname"
done
