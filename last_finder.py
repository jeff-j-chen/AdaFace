import os
from tqdm import tqdm
with open("names.txt", "r") as f:
    names = f.readlines()
files = os.listdir('faces_google')
missing_names = []
for i in tqdm(range(len(names))):
    name = names[i]
    fname = "_".join(name.strip().split(" "))
    if fname+"_2.png" not in files and fname+"_2.jpg" not in files:
        counter += 1
        if counter > 10:
            print(name)
            print(i-10)
            break
    else:
        counter = 0