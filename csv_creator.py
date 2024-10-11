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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


adaface_models = {
    'ir_101':"pretrained/adaface_ir101_webface4m.ckpt",
}

def load_pretrained_model(architecture='ir_101', device='gpu'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location=device)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    model.to(device)  # Move model to the specified device (GPU or CPU)
    return model

def to_input(pil_rgb_image, device='gpu'):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor(np.array([brg_img.transpose(2,0,1)])).float()
    return tensor.to(device)  # Move tensor to the specified device (GPU or CPU)


csv_lock = threading.Lock()

def process_feature(fname, model, device, test_image_path):
    path = os.path.join(test_image_path, fname)
    try:
        status, aligned_rgb_img = align.get_aligned_face(path)
    except:
        return 1  # Skipping failed reads
    if status == 0:
        bgr_tensor_input = to_input(aligned_rgb_img, device)
        f, _ = model(bgr_tensor_input)

        feature_string = ','.join(map(str, f[0].cpu().detach().numpy().flatten()))
        name = os.path.splitext(fname)[0]
        
        # Ensure thread-safe CSV writing
        with csv_lock:
            with open("output_google/features_google.csv", mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([name, feature_string])
        torch.cuda.synchronize()
        return 0
    else:
        torch.cuda.synchronize()
        return 1  # Skipping missing faces

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using device {device}')
    model = load_pretrained_model('ir_101', device=device)
    feature, norm = model(torch.randn(2,3,112,112).to(device))
    test_image_path = '/media/jeff/Seagate/adaface/faces_google'
    
    # Read names from file
    with open("names.txt", "r") as f:
        names = f.readlines()

    tasks = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        print("loading faces..")
        for i in tqdm(range(len(names))):
            if i < 20556: continue
            name = '_'.join(names[i].strip().split(' '))
            for i in range(1, 4):                
                tasks.append(
                    executor.submit(
                            process_feature, f"{name}_{i}.jpg", model, device, test_image_path
                        )
                    )
        
        for i in (pbar := tqdm(range(len(tasks)), desc="processingg faces..")):
            for future in as_completed(tasks):
                future.result()
                pbar.update(1)

if __name__ == '__main__':
    main()