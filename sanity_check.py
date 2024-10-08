import net
import torch
import os
from face_alignment import align
import numpy as np
import torch.nn.functional as F
import cv2
import csv
from sklearn.metrics.pairwise import cosine_similarity
import time



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

def load_features_csv(csv_path):
    features = []
    names = []
    with open(csv_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            name = row[0]
            feature = np.array([list(map(float, row[1].split(',')))])
            names.append(name)
            features.append(feature.reshape(1, -1))

    return names, np.array(features)

if __name__ == '__main__':
    start_time = time.time()
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    
    names, features = load_features_csv("output/features.csv")
    print(f"going for {len(features)} iterations")
    for i in range(1, len(features)):
        sim_1 = cosine_similarity(features[i], features[i])
        sim_diff = cosine_similarity(features[i], features[i-1])
        print(f"compairons between features of {names[i]} and {names[i-1]}")
        print(sim_1)
        print(sim_diff)