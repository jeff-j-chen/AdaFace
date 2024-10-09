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

def process_feature(fname):
    path = os.path.join(test_image_path, fname)
    try:
        status, aligned_rgb_img = align.get_aligned_face(path)
    except:
        failed_reads.append(fname + " (bad image)")
        return 1
    if status == 0:
        bgr_tensor_input = to_input(aligned_rgb_img)
        f, _ = model(bgr_tensor_input)

        with open("output/features.csv", mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            name = os.path.splitext(fname)[0]
            feature_string = ','.join(map(str, f[0].detach().numpy().flatten()))
            writer.writerow([name, feature_string])
        return 0
    else:
        failed_reads.append(fname + "(missing face)")
        return 1

if __name__ == '__main__':
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))

    test_image_path = '/media/jeff/Seagate/adaface/faces'
    prev_name = ""
    to_process = []
    missing_names = []
    failed_reads = []
    with open("names.txt", "r") as f:
        names = f.readlines()
    for i in (pbar := tqdm(range(len(names)))):
        name = '_'.join(names[i].strip().split(' '))
        pot_missing = False
        pbar.set_description(f"processing for {name.strip()}")
        for i in range(1, 4):
            fname_png = f"{name}_{i}.png"
            fname_jpg = f"{name}_{i}.jpg"
            res = 1
            if os.path.isfile(os.path.join(test_image_path, fname_png)):
                res = process_feature(fname_png)
            elif os.path.isfile(os.path.join(test_image_path, fname_jpg)):
                res = process_feature(fname_jpg)
            else:
                if i == 1 or pot_missing:
                    missing_names.append(name)
                break
            if i == 1 and res == 1:
                # on the first read, if the image read is a failure, mark this as a potential missing name
                pot_missing = True
            elif res == 0:
                pot_missing = False
    
    with open("output/missing_names.txt", "w") as f:
        f.writelines(n + "\n" for n in missing_names)
    with open("output/failed_reads.txt", "w") as f:
        f.writelines(n + "\n" for n in failed_reads)
