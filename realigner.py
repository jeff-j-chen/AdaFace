import os
import sys

# order the files by leading number, trim all excess
files = os.listdir("faces_google/")
files.sort(key=lambda x: int(x.split('_')[0]))
seen = set()
culled_files = []
for file in files:
    leading_number = int(file.split('_')[0])
    if leading_number not in seen:
        culled_files.append(file)
        seen.add(leading_number)

# 'blacklist' all consecutive names, since they seem to have caused errors
with open("names.txt", "r") as f:
    names_txt = f.readlines()
names_txt = [n.strip() for n in names_txt]
blacklist = []
for i in range(1, len(culled_files)):
    if names_txt[i] == names_txt[i-1]:
        blacklist.append(i+1)

# find missing names and blacklist them, too
only_names = [" ".join(f.split("_")[1:-1]) for f in culled_files]
for i in range(len(culled_files)):
    if names_txt[i] not in only_names:
        blacklist.append(i+1)
        print(f" added {i+1}")

def find_index_by_number(files, target_number):
    return next(
        (i for i, file in enumerate(files) if int(file.split('_')[0]) == target_number),
        -1  # Return None if the number is not found
    )

for i in range(1, 20567):
    found_i = find_index_by_number(culled_files, i+1)
    if i+1 in blacklist: continue
    if found_i == -1:
        print(f"error attempting to find culled file number {i+1}")
        break
    global_i = culled_files[found_i].split("_")[0]
    name = only_names[found_i]
    print(f"checking {names_txt[i]} vs {name}")
    if names_txt[i] != name:
        print(f"broke at file i {global_i}, loop iteration {i}")
        break