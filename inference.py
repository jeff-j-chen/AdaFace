import net
import torch
import os
from face_alignment import align
import numpy as np
import torch.nn.functional as F
import cv2
import csv
from sklearn.metrics.pairwise import cosine_distances
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
            name, feature_str = row
            feature = np.fromstring(feature_str[1:-1], sep=' ')
            names.append(name)
            features.append(feature)

    return names, np.array(features)

if __name__ == '__main__':
    start_time = time.time()
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    # path = "/media/jeff/Seagate/adaface/faces_temp/adley rutschman.png"
    # path = "/media/jeff/Seagate/Downloads/all_cards/rutchsman.jpg"
    # path = "/media/jeff/Seagate/Downloads/all_cards/noren.jpg"
    # path = "/media/jeff/Seagate/Downloads/all_cards/blylven.jpg"
    path = "/media/jeff/Seagate/Downloads/all_cards/ichiro.jpg"
    status, aligned_rgb_img = align.get_aligned_face(path)
    if status != 0:
        print("no face detected!")
        exit()
    cv2.imshow("test", cv2.cvtColor(np.array(aligned_rgb_img), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    bgr_tensor_input = to_input(aligned_rgb_img)
    feature, _ = model(bgr_tensor_input)
    print(f"load time: {time.time()-start_time}")
    
    start_time = time.time()
    names, features = load_features_csv("features.csv")
    distances = cosine_distances(feature.detach().numpy().reshape(1, -1), features)
    most_similar_indices = distances.argsort()[0][:5]
    print(f"search time: {time.time()-start_time}")
    print("Indices of the top 5 most similar features:")
    print(most_similar_indices)
    for i in most_similar_indices:
        print(f"{i}: {names[i]}")
