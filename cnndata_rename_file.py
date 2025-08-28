import os,json,re

folder = r"C:\Users\ABC\Documents\cnndata\images"
text_file = r"C:\Users\ABC\Documents\cnndata\labels.json"
name_list = []
data = []
for idx,fname in enumerate(os.listdir(folder)):
    newname = fname.replace("_crop_pyttext","")
    name_list.append(newname)
    if "_pyttext_" not in fname:
        continue  # skip files without marker

    # # split once at "_pyttext_"
    # before, after = fname.split("_pyttext_", 1)
    old_path = os.path.join(folder, fname)
    # # split once at "_pyttext_"
    # before, after = fname.split("_pyttext_", 1)

    # # prepend index to make names unique
    # new_name = f"{idx}_{after}"
    
    new_path = os.path.join(folder, newname)

    # rename
    
    # print(f"Renaming: {fname} -> {new_name}")
    if old_path != new_path:
        if os.path.exists(new_path):  # only rename if target doesn't exist
            os.remove(new_path)
        os.rename(old_path, new_path)
        

for fname in name_list:
    if "labels"  in fname:
        continue  # skip files without marker
    # fname = fname.replace(" ","_")
    chunks = fname.split("_")
    annotation = chunks[1]
    annotation = annotation.replace("jpy","￥")
    # annotation = annotation.replace(" ","")
    cleaned = re.sub(r"[A-Za-z\s]", "", annotation)
    # cleaned = re.sub(r"[^0-9\-]", "", annotation)
    print(cleaned)
    if cleaned != "":
        data.append({
            "filename": fname,
            "annotation": cleaned
        })

with open(text_file, "w", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=4))
print(len(name_list))
