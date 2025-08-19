import os

folder = r"C:\Users\ABC\Documents\receiptYOLOProject\cnndata\images"
text_file = r"C:\Users\ABC\Documents\receiptYOLOProject\cnndata\labels.txt"
name_list = []
for idx,fname in enumerate(os.listdir(folder)):
    name_list.append(fname)
#     if "_pyttext_" not in fname:
#         continue  # skip files without marker

#     # split once at "_pyttext_"
#     before, after = fname.split("_pyttext_", 1)
#     old_path = os.path.join(folder, fname)
#     # split once at "_pyttext_"
#     before, after = fname.split("_pyttext_", 1)

#     # prepend index to make names unique
#     new_name = f"{idx}_{after}"
#     new_path = os.path.join(folder, new_name)

#     # rename
#     print(f"Renaming: {fname} -> {new_name}")
#     os.rename(old_path, new_path)

with open(text_file, "w", encoding="utf-8") as f:
    for fname in name_list:
        if "labels"  in fname:
            continue  # skip files without marker
        chunks = fname.split("_")
        annotation = chunks[1]
        line = f"{fname},{annotation}\n"
        f.write(line)

