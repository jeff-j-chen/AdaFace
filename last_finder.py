# import os
# from tqdm import tqdm
# with open("names.txt", "r") as f:
#     names = f.readlines()
# files = os.listdir('faces_google')
# missing_names = []
# for i in tqdm(range(len(names))):
#     name = names[i]
#     fname = "_".join(name.strip().split(" "))
#     if fname+"_2.png" not in files and fname+"_2.jpg" not in files:
#         counter += 1
#         if counter > 10:
#             print(name)
#             print(i-10)
#             break
#     else:
#         counter = 0

import csv
import numpy as np
def load_features_dict_csv(csv_path):
    interest = ["joel zumaya", "guillermo zuniga", "mike zunino", "bob zupcic", "frank zupo", "paul zuvella", "george zuverink", "dutch zwilling", "tony zych"]
    loaded = {}
    with open(csv_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name = row[0]
            loaded[name] = ""
            if name in interest:
                print(name)

    return loaded
loaded = load_features_dict_csv("output_google/features_google.csv")