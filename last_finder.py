import os
from tqdm import tqdm
with open("names.txt", "r") as f:
    names = f.readlines()
files = os.listdir('faces')
missing_names = []
for i in tqdm(range(len(names))):
    name = names[i]
    fname = "_".join(name.strip().split(" "))
    if fname+"_1.png" not in files and fname+"_1.jpg" not in files:
        missing_names.append(name)
with open("missing_afterscan.txt", "w") as f:
    f.writelines(missing_names)