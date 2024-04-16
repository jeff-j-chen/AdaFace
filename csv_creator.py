import net
import torch
import os
import csv
from face_alignment import align
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from sklearn.metrics.pairwise import cosine_similarity


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

if __name__ == '__main__':
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = 'faces_temp'
    features = []
    names = []
    for fname in tqdm(sorted(os.listdir(test_image_path))):
        print(fname)
        path = os.path.join(test_image_path, fname)
        status, aligned_rgb_img = align.get_aligned_face(path)
        if status == 0:
            bgr_tensor_input = to_input(aligned_rgb_img)
            feature, _ = model(bgr_tensor_input)
            names.append(fname)
            features.append(feature)

    

    # Define the path to save the CSV file
    csv_path = "features.csv"

    # Open the CSV file in write mode
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['name', 'feature'])
        for fname, feature in tqdm(zip(names, features)):
            name = os.path.splitext(fname)[0]
            writer.writerow([name, feature[0].detach().numpy()])

