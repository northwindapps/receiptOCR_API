from PIL import Image
import imagehash, os

folder = r"C:\Users\ABC\Documents\cnndata\images"
pmap = {}  # mapping from hash -> list of filenames

for fname in os.listdir(folder):
    try:
        full_path = os.path.join(folder, fname)
        h = str(imagehash.average_hash(Image.open(full_path).convert('L'), hash_size=16))
        pmap.setdefault(h, []).append(fname)
    except Exception as e:
        print(f"Skipping {fname}: {e}")

near_dups = {h: files for h, files in pmap.items() if len(files) > 1}
print("Perceptual dup groups:", len(near_dups))

for h, files in near_dups.items():
    print(f"Hash: {h} -> {files}")


# Keep only one image per duplicate group
# for h, files in near_dups.items():
#     # Keep the first file, remove the rest
#     for f in files[1:]:
#         path_to_remove = os.path.join(folder, f)
#         if os.path.exists(path_to_remove):
#             os.remove(path_to_remove)
#             print(f"Removed duplicate: {f}")

