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
    path = "/media/jeff/Seagate/adaface/faces_temp/bert blyleven (old).png"
    name = os.path.splitext(os.path.basename(path))[0]
    status, aligned_rgb_img = align.get_aligned_face(path)
    if status == 0:
        bgr_tensor_input = to_input(aligned_rgb_img)
        feature, _ = model(bgr_tensor_input)
        with open("f.txt", "w") as f:
            f.write(f'{name},"{feature.detach().numpy()[0]}"')
        print(f"saved to f.txt")
    else:
        print(f"no face found")