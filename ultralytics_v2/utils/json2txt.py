import json
import base64
import numpy as np
import cv2
import glob
import os
from sklearn.model_selection import train_test_split


def get_image(img_b64):
    im_bytes = base64.b64decode(img_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def yolo_label(data,img_w,img_h,classes):
    yolo_obj_list = []
    for shapes in data:
        retval = [list(classes.values()).index(shapes['label'])]
        for i in shapes['points']:
            i[0] = round(float(i[0]) / img_w, 6)
            i[1] = round(float(i[1]) / img_h, 6)
            retval.extend(i)
        retval_str = ' '.join(map(str,retval))
        yolo_obj_list.append(retval_str)
    return yolo_obj_list

def json2txt(path,class_label):
    path = path + '/images'

    for file in glob.glob(path+'/*.json'):
        filename = os.path.basename(file)
        data = json.load(open(file))
        img = get_image(data['imageData'])
        img_h, img_w, _ = img.shape
        cv2.imwrite(file.replace('.json','.png'),img)

        txt_data = yolo_label(data['shapes'],img_w,img_h,class_label)
        if not os.path.exists(path.replace('/images','/labels')):
            os.mkdir(path.replace('/images','/labels'))

        with open(file.replace('.json','.txt').replace('/images','/labels'), 'w') as f:
            f.write('\n'.join(txt_data))
            f.close()

    files = glob.glob(path+'/*.png')
    lst_train, lst_val = train_test_split(files, test_size=0.1)

    #creating the txt files containing the file name 
    for lst_yolo in lst_train:
        with open(os.path.dirname(path)+'/train.txt', 'a+') as f1:
            f1.write(lst_yolo+"\n")

    for lst_yolo in lst_val:
        with open(os.path.dirname(path)+'/val.txt', 'a+') as f2:
            f2.write(lst_yolo+"\n")