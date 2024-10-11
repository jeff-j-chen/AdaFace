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

def pad_images(img1, img2):
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    max_height = max(height1, height2)

    if height1 < max_height:
        padding = max_height - height1
        img1 = cv2.copyMakeBorder(img1, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if height2 < max_height:
        padding = max_height - height2
        img2 = cv2.copyMakeBorder(img2, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.hstack((img1, img2))

def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img

if __name__ == '__main__':
    model = load_pretrained_model('ir_101')
    feature, norm = model(torch.randn(2,3,112,112))
    
    features = load_features_dict_csv("output_google/features_google.csv")
    test_image_path = '/media/jeff/Seagate/adaface/faces_google'
    with open("names.txt", "r") as f:
        names = f.readlines()
    thresh = 0.60
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
        valid_n = []
        for i in range(1, 4):
            if f"{name}_{i}" in features: 
                valid_n.append(f"{name}_{i}")
                valid_f.append(features[f"{name}_{i}"])
        images = []
        for filename in [f"{name}_1", f"{name}_2", ]:
            file_test = os.path.join(test_image_path, filename+'.jpg')
            if os.path.isfile(file_test):
                images.append(cv2.imread(file_test))

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

            combined = combined = pad_images(images[0], pad_images(images[1], images[2]))
            if max(counts) == 0: # no faces are similar
                no_similar.append(name)
                res.append(f"{name}** (no similar, 3)")
                cv2.putText(combined, f"({str(sim_12)}, {str(sim_13)}, {str(sim_23)}) {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
            elif max(counts) == 1: # only 1 pair of faces are similar
                dissimilar.append(name)
                res.append(f"{name}** (1 pair similar, 3)")
                cv2.putText(combined, f"({str(sim_12)}, {str(sim_13)}, {str(sim_23)}) {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, lineType=cv2.LINE_AA)
            else: # 2 or more similar faces
                majority.append(name)
                res.append(f"{name} (majority)")
                cv2.putText(combined, f"({str(sim_12)}, {str(sim_13)}, {str(sim_23)}) {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        if len(valid_f) == 2:
            sim_12 = round(cosine_similarity(valid_f[0], valid_f[1])[0][0], 3)
            combined = pad_images(images[0], images[1])
            if sim_12 > thresh:
                majority.append(name)
                res.append(f"{name} ({sim_12})")
                combined = resize_img(combined, 900)
                cv2.putText(combined, f"({str(sim_12)}) {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            else:
                dissimilar.append(name)
                res.append(f"{name}** ({sim_12})")
                combined = resize_img(combined, 900)
                cv2.putText(combined, f"({str(sim_12)}) {name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.imshow('Stitched Image with Text', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(f"output_google_stitched/{name}.jpg", combined)

        elif len(valid_f) == 1:
            only_1.append(name)
            res.append(f"{name}* (single)")

        else:
            missing_names.append(name)
            res.append(f"{name}** (missing)")
        
        
    with open("output_google_filtered/missing_names.txt", "w") as f:
        f.writelines(n + "\n" for n in missing_names)
    with open("output_google_filtered/only_1.txt", "w") as f:
        f.writelines(n + "\n" for n in only_1)
    with open("output_google_filtered/majority.txt", "w") as f:
        f.writelines(n + "\n" for n in majority)
    with open("output_google_filtered/dissimilar.txt", "w") as f:
        f.writelines(n + "\n" for n in dissimilar)
    with open("output_google_filtered/no_similar.txt", "w") as f:
        f.writelines(n + "\n" for n in no_similar)
    with open("output_google_filtered/all.txt", "w") as f:
        f.writelines(n + "\n" for n in res)
