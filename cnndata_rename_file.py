import os,json,re

folder = r"C:\Users\ABC\Documents\receiptYOLOProject\2025_0822_bad_cnndata\cnndata\images"
text_file = r"C:\Users\ABC\Documents\receiptYOLOProject\cnndata\labels.json"
name_list = []
data = []
for idx,fname in enumerate(os.listdir(folder)):
    name_list.append(fname)
    # if "_pyttext_" not in fname:
    #     continue  # skip files without marker

    # # split once at "_pyttext_"
    # before, after = fname.split("_pyttext_", 1)
    # old_path = os.path.join(folder, fname)
    # # split once at "_pyttext_"
    # before, after = fname.split("_pyttext_", 1)

    # # prepend index to make names unique
    # new_name = f"{idx}_{after}"
    # new_path = os.path.join(folder, new_name)

    # # rename
    # print(f"Renaming: {fname} -> {new_name}")
    # os.rename(old_path, new_path)

for fname in name_list:
    if "labels"  in fname:
        continue  # skip files without marker
    chunks = fname.split("_")
    annotation = chunks[3]
    cleaned = re.sub(r"[A-Za-z\s]", "", annotation)
    print(cleaned)
    data.append({
        "filename": fname,
        "annotation": cleaned
    })

with open(text_file, "w", encoding="utf-8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=4))

