import net
import torch
import os
from face_alignment import align
import numpy as np
import torch.nn.functional as F
import cv2
import csv
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm



adaface_models = {
    'ir_101':"pretrained/adaface_ir101_webface4m.ckpt",
}

def load_pretrained_model(architecture='ir_101'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(np.array([brg_img.transpose(2,0,1)])).float()
    return tensor

def load_features_dict_csv(csv_path):
    loaded = {}
    with open(csv_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name = row[0]
            feature = np.array([list(map(float, row[1].split(',')))])
            loaded[name] = feature

    return loaded

if __name__ == '__main__':
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    
    features = load_features_dict_csv("output/features.csv")
    with open("names.txt", "r") as f:
        names = f.readlines()
    thresh = 0.66
    no_similar = []
    dissimilar = []
    majority = []
    only_1 = []
    missing_names = []
    res = []
    for i in (pbar := tqdm(range(len(names)))):
        name = '_'.join(names[i].strip().split(' '))
        pbar.set_description(f"processing for {name.strip()}")
        valid_f = []
        for j in range(3):
            if f"{name}_{j}" in features: valid_f.append(features[f"{name}_{j}"])
        if len(valid_f) == 3:
            pairs = []
            sim_12 = cosine_similarity(valid_f[0], valid_f[1])
            sim_13 = cosine_similarity(valid_f[0], valid_f[2])
            sim_23 = cosine_similarity(valid_f[1], valid_f[2])
            if sim_12 > thresh: pairs.extend([0, 1])
            if sim_13 > thresh: pairs.extend([0, 2])
            if sim_23 > thresh: pairs.extend([1, 2])
            counts = [0, 0, 0]
            for p in pairs:
                counts[p] += 1
            if max(counts) == 0: # no faces are similar
                no_similar.append(name)
                res.append(f"{name}** (no similar, 3)")
            elif max(counts) == 1: # only 1 pair of faces are similar
                dissimilar.append(name)
                res.append(f"{name}** (1 pair similar, 3)")
            else: # 2 or more similar faces
                majority.append(name)
                res.append(f"{name} (majority)")

        elif len(valid_f) == 2:
            sim_12 = cosine_similarity(valid_f[0], valid_f[1])
            if sim_12 > thresh:
                majority.append(name)
                res.append(f"{name} (1 pair similar, 2)")
            else:
                dissimilar.append(name)
                res.append(f"{name}** (no similar, 2)")

        elif len(valid_f) == 1:
            only_1.append(name)
            res.append(f"{name}* (single)")

        else:
            missing_names.append(name)
            res.append(f"{name}** (missing)")
        
        
    with open("output_filtered/missing_names.txt", "w") as f:
        f.writelines(n + "\n" for n in missing_names)
    with open("output_filtered/only_1.txt", "w") as f:
        f.writelines(n + "\n" for n in only_1)
    with open("output_filtered/majority.txt", "w") as f:
        f.writelines(n + "\n" for n in majority)
    with open("output_filtered/dissimilar.txt", "w") as f:
        f.writelines(n + "\n" for n in dissimilar)
    with open("output_filtered/no_similar.txt", "w") as f:
        f.writelines(n + "\n" for n in no_similar)
    with open("output_filtered/all.txt", "w") as f:
        f.writelines(n + "\n" for n in res)
