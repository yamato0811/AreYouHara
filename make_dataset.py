from keras_facenet import FaceNet
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageOps

from utils import crop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

face_dir = "hara-imgs"

embeddings = [] # 顔ベクトル（顔特徴量）
imgs = []
save_paths = []
facenet = FaceNet() # FaceNetモデル

files = os.listdir(face_dir) # ディレクトリ のファイルリストを取得
for i,file in enumerate(tqdm(files)):
    file_path = os.path.join(face_dir, file)

    image = Image.open(file_path).convert('RGB')
    image = ImageOps.exif_transpose(image)  # 画像を適切な向きに補正する
    img_array = np.array(image)

    extracts = facenet.extract(img_array)
    if len(extracts) < 1:
        continue
    max_extract = max(extracts, key=lambda x:x['box'][2]*x['box'][3])
    embeddings.append(max_extract['embedding']) 
    
    save_path = f'hara_data/imgs/hara_{i}.npy'
    save_paths.append(save_path)

    img_array = crop(img_array, max_extract['box'])
    np.save(save_path, img_array)

np.save('hara_data/img_paths', np.array(save_paths))
np.save('hara_data/embeddings', np.array(embeddings))
