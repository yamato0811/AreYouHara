from PIL import Image

# 距離:2~0, 類似度:0~100
def distance_to_similarity(dis, min_dis=0.25, max_dis=1.0):
    if dis < min_dis:
        similarity = 100
    elif dis > max_dis:
        similarity = 0
    else:
        similarity = (100/(min_dis-max_dis))*dis + 100*max_dis/(max_dis-min_dis)
    return similarity

def crop(img, box):
    x,y,width,hight = box
    return img[y:y+hight, x:x+width]

def resize(img, width=960, height=540):
    if img.width > width or img.height > height:
        img = img.resize((img.width//2,img.height//2),Image.BICUBIC)
    return img

def _keepAspectResize(image, size):

    # サイズを幅と高さにアンパック
    width, height = size

    # 矩形の幅と画像の幅の比率を計算
    x_ratio = width / image.width

    # 矩形の高さと画像の高さの比率を計算
    y_ratio = height / image.height

    # 画像の幅と高さ両方に小さい方の比率を掛けてリサイズ後のサイズを計算
    if x_ratio < y_ratio:
        resize_size = (width, round(image.height * x_ratio))
    else:
        resize_size = (round(image.width * y_ratio), height)

    # リサイズ後の画像サイズにリサイズ
    resized_image = image.resize(resize_size)

    return resized_image

def adjust_img_margin(uimg, pimg, size=(500,625)): # uploaded_img, picked_hara_img
    uimg = Image.fromarray(uimg)
    pimg = Image.fromarray(pimg)
    uimg = _keepAspectResize(uimg, size)
    pimg = _keepAspectResize(pimg, size)
    
    u_width, u_height = uimg.size
    p_width, p_height = pimg.size

    TRANSPARENT = (0,0,0,0)  # 透明色

    # y-axis
    if u_height > p_height:
        pane = Image.new('RGBA', (p_width, u_height), TRANSPARENT)
        pane.paste(pimg, (0, int((u_height-p_height)/2)))
        pimg = pane
    elif p_height > u_height:
        pane = Image.new('RGBA', (u_width, p_height), TRANSPARENT)
        pane.paste(uimg, (0, int((p_height-u_height)/2)))
        uimg = pane
    
    # x-axis
    if u_width > p_width:
        pane = Image.new('RGBA', (u_width, p_height), TRANSPARENT)
        pane.paste(pimg, ((int((u_width-p_width)/2)), 0))
        pimg = pane
    elif p_width > u_width:
        pane = Image.new('RGBA', (p_width, u_height), TRANSPARENT)
        pane.paste(uimg, ((int((p_width-u_width)/2)), 0))
        uimg = pane

    return uimg, pimg
