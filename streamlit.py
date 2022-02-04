from torch import minimum
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras_facenet import FaceNet

from utils import *
from dataloader import Dataloader


st.title('Similarity with Hara')
st.write("早速あなたの顔の画像をアップロードしてみましょう")

uploaded_file = st.file_uploader('Choose a image file')

facenet = FaceNet()
dataloader = Dataloader()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = ImageOps.exif_transpose(image)  # 画像を適切な向きに補正する
    print(image.size)

    if image.width > 960 or image.height >540:
         #image = image.resize((960,540),Image.BICUBIC)
          image = image.resize((image.width//2,image.height//2),Image.BICUBIC)
          print(image.size)


    img_array = np.array(image)

    extracts = facenet.extract(img_array)

    # 顔が画像中に存在しないとき
    if len(extracts) < 1:
        st.header('Face detection faild')
        st.image(
            img_array, 
            use_column_width=True
        )
    else:
        max_extract = max(extracts, key=lambda x:x['box'][2]*x['box'][3]) # もっとも大きい顔を取得

        embed_img = max_extract['embedding']

        distances = [(facenet.compute_distance(e, embed_img), p) for e, p in dataloader]
        distance, path = min(distances)
        similarity = distance_to_similarity(distance)
        st.subheader(f'Similarity : {round(similarity)} %')
        st.text(f'distance : {distance}')

        img_array = crop(img_array, max_extract['box'])

        col1, col2 = st.columns(2)

        with col1:
            st.header("Upload Image")
            st.image(
                img_array, 
                use_column_width=True
            )

        with col2:
            st.header("Most Similar Hara")
            st.image(
                dataloader.load_img(path), 
                use_column_width=True
            )