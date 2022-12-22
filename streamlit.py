from distutils.command.upload import upload
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from keras_facenet import FaceNet

from utils import distance_to_similarity, crop, resize, adjust_img_margin
from dataloader import Dataloader

st.set_page_config(
    page_title="AreYouHara"
)

@st.cache(allow_output_mutation=True)
def load_facenet():
    return FaceNet()
@st.cache
def load_dataloader():
    return Dataloader()

# CSSの読み込み
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    st.title('Are You Hara')
    st.write('あなたの顔がハラかどうか判定します  \n 早速あなたの顔の画像をアップロードしてみましょう')
    st.markdown('[Qiita]()  [GitHub](https://github.com/yamato0811/AreYouHara)')
with col2:
    hara_image = Image.open('hara_data/Hara.JPG')
    hara_image = ImageOps.exif_transpose(hara_image)
    st.image(hara_image, caption='ハラ', width=100)

uploaded_file = st.file_uploader('Choose a image file', type=['jpg','jpeg','png'])

facenet = load_facenet()
dataloader = load_dataloader()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = ImageOps.exif_transpose(image)  # 画像を適切な向きに補正する
    image = resize(image)

    img_array = np.array(image)
    extracts = facenet.extract(img_array) # 顔領域を抽出

    if len(extracts) < 1:
    # 顔が画像中に存在しないとき
        st.header('Face detection faild')
        st.image(
            img_array, 
            use_column_width=True
        )
    else:
        max_extract = max(extracts, key=lambda x:x['box'][2]*x['box'][3]) # もっとも大きい顔を取得(複数の顔が存在する場合がある)
        embed_img = max_extract['embedding'] # 512次元の特徴を取得

        distances = [(facenet.compute_distance(e, embed_img), p) for e, p in dataloader] # hara dataset内のすべての画像と距離を計算
        distance, path = min(distances)
        similarity = distance_to_similarity(distance) # 距離を類似度に変換
        
        # 判定結果
        if similarity == 100:
            st.text('You are HARA🎉🎉🎉')
        else:
            st.text('You are not HARA😞😞😞')
        st.text(f'Similarity : {round(similarity)} %')

        # Twitterシェアボタン
        twitter_btn = f"""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>
        <div class="tweet-btn">
            <a href="https://twitter.com/share?text=私とハラの類似度は%20{ str(round(similarity)) }%25%20でした%0a&url=https://areyouhara.streamlit.app/&hashtags=AreYouHara" id="tweet">
                <i class="fa-brands fa-twitter"></i>
                <p>ツイート</p>
            </a>
        </div>
        """
        st.write(twitter_btn, unsafe_allow_html=True)

        img_array = crop(img_array, max_extract['box'])
        uploaded_img, picked_img = adjust_img_margin(img_array, dataloader.load_img(path)) # 表示の際に適切なマージンがつくよう調整

        col1, col2 = st.columns(2)
        with col1:
            st.write("Upload Image")
            st.image(
                uploaded_img, 
                use_column_width=True
            )
        with col2:
            st.write("Most Similar Hara")
            st.image(
                picked_img, 
                use_column_width=True
            )
