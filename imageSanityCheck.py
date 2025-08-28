import random, cv2,os

folder = r"C:\Users\ABC\Documents\cnndata\images"
image_paths=[]
texts=[]
for fname in os.listdir(folder):
    full_path = os.path.join(folder, fname)
    image_paths.append(full_path)
    chunks = fname.split("_")
    annotation = chunks[1]
    annotation = annotation.replace("jpy","￥")
    texts.append(annotation)

for p, lbl in random.sample(list(zip(image_paths, texts)), 10):
    print("GT:", lbl)
    im = cv2.imread(p)
    if im is None: 
        print("can't open", p); continue
    cv2.imshow("sample", cv2.resize(im, (im.shape[1], im.shape[0])))
    cv2.waitKey(2500)  # or save to disk if headless
cv2.destroyAllWindows()
