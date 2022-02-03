
# 距離:2~0, 類似度:0~100
def distance_to_similarity(dis, min_dis=0.25, max_dis=1.0):
    if dis < min_dis:
        similarity = 100
    elif dis > max_dis:
        similarity = 0
    else:
        print(dis)
        similarity = (100/(min_dis-max_dis))*dis + 100*max_dis/(max_dis-min_dis)
    return similarity

def crop(img, box):
    x,y,width,hight = box
    return img[y:y+hight, x:x+width]