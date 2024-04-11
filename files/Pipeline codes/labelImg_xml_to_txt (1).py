import os
import copy
import random
import glob
import sys

import cv2
import numpy as np
import xml.etree.ElementTree as ET


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    x = round(x, 6)
    w = w * dw
    w = round(w, 6)
    y = y * dh
    y = round(y, 6)
    h = h * dh
    h = round(h, 6)
    return x, y, w, h


def add_defects(image, b, cls, file_name):
    if cls == "rod":
        temp_image = copy.deepcopy(image)
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = int(b[0]), int(b[2]), int(b[1]), int(b[3])
        # cropped_image = temp_image[y1:y2, x1:x2]
        defect_image = cv2.imread("D:/Work/NBM/defects/3-set-03.png", cv2.IMREAD_UNCHANGED)
        defect_height, defect_width, _ = defect_image.shape
        k1 = cv2.imread("D:/Work/NBM/defects/3-set-03.png", 0)
        k = defect_image[:, :, 3]
        # cv2.imshow("mask", k)
        defect_x1 = random.randint(x1, int(x1+x2/2))
        defect_y1 = random.randint(y1, (y2-defect_height))
        defect_x2 = defect_x1 + defect_width
        defect_y2 = defect_y1 + defect_height

        logo1_x1, logo1_y1, logo1_x2, logo1_y2 = defect_x1, defect_y1, defect_x2, defect_y2

        destination1 = temp_image[logo1_y1:logo1_y2, logo1_x1:logo1_x2]

        img1_bg = cv2.bitwise_and(k1, k)
        # cv2.imshow("image___", img1_bg)
        img1_bg[(img1_bg > 135) & (img1_bg < 255)] = 92
        # img1_bg = img1_bg[np.where((img1_bg == [0, 0]).all(axis=1))] = [132, 132]
        new = cv2.addWeighted(destination1, 1, img1_bg, 0.9, 0.1)
        # new = cv2.add(destination1, img1_bg)

        temp_image[logo1_y1:logo1_y2, logo1_x1:logo1_x2] = new

        # cv2.imshow("---", new1)

        # cv2.imshow("result1", new)
        # cv2.waitKey()

        # cv2.imwrite("temp/temp_3/"+str(file_name)+".png", temp_image)
        return temp_image


def draw_rectangle(image, b, cls):
    x1, y1, x2, y2 = int(b[0]), int(b[2]), int(b[1]), int(b[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
    cv2.putText(image, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


class Voc_Xml:

    def __init__(self, class_names=None, classes_to_combine=None, image_files_path="/images", xml_files_path="/xmls",
                 labels_save_path="/labels"):
        if class_names is None:
            class_names = []
        # ext = ['.png', '.jpg', '.jpeg', ".bmp"]
        # self.image_files_path = []
        self.image_files_path = image_files_path
        self.xml_files_path = glob.glob(xml_files_path + "/*.xml")
        self.labels_save_path = labels_save_path
        self.class_names = class_names
        self.classes_to_combine = classes_to_combine
        self.file_end = [".jpg", ".JPG", ".jpeg", ".png", ".bmp"]
        os.makedirs(labels_save_path, exist_ok=True)
        if len(self.class_names) == 0:
            print("Empty list in the classes please check the class_names list in the code")
            sys.exit()

        if len(self.xml_files_path) == 0:
            print("Check the xml folder path and check whether the xml files are present in the directory or not")
            sys.exit()

    def convert_xml_txt(self):
        """
        convernts xml file to txt files and save it into the directory
        :return: none
        """
        for xml_path in self.xml_files_path:
            basename = os.path.basename(xml_path)
            basename_no_ext = os.path.splitext(basename)[0]
            in_file = open(xml_path)
            out_file = open(os.path.join(self.labels_save_path, basename_no_ext + '.txt'), 'a')
            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            
            for file_end in self.file_end:
                if os.path.exists(os.path.join(self.image_files_path, basename_no_ext + str(file_end))):
                    image_file_path = os.path.join(self.image_files_path, basename_no_ext + str(file_end))
            print(image_file_path)
            if os.path.exists(image_file_path):
                image = cv2.imread(image_file_path)
                h, w, _ = image.shape
                w = int(w)
                h = int(h)
                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    cls = cls.lower()
                    # cls = "defect" if cls == "rod" else cls
                    if self.classes_to_combine is not None:
                        print(cls)
                        if cls in self.classes_to_combine:
                            cls = "vehicle"
                    if cls in self.class_names:
                        cls_id = self.class_names.index(cls.lower())
                        xmlbox = obj.find('bndbox')
                        # print(cls_id)
                        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                             float(xmlbox.find('ymin').text),
                             float(xmlbox.find('ymax').text))
                        # save_images(image, b, cls)
                        # temp_image = add_defects(image, b, cls, basename_no_ext)
                        image = draw_rectangle(image, b, cls)
                        bb = convert((w, h), b)
                        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
                    else:
                        print(
                            f"The class name in xml is {cls} which might not be found in the list entered or typo in the class name in xml")
                cv2.imshow("image", cv2.resize(image, (640, 640)))
                # cv2.imshow("image", temp_image)
                key = cv2.waitKey(1)
                if key == 27:
                    break
            else:
                print(f"Image not found {image_file_path}")


# classes = ["vehicle", "np"]
# classes = ["auto", "bus", "car", "hmv", "motorcycle", "np", "tractor","truck"]
classes = ["auto", "bus", "car", "motorcycle", "np", "truck"]
# classes_to_combine = ["car", "auto", "motorcycle", "truck", "bus", "hmv"]
# classes = ["human"]
# classes = ["bar", "defect"]
# classes = ["back", "front", "vehicle"]

obj = Voc_Xml(class_names=classes, classes_to_combine=None, 
              image_files_path="D:/Work/data/Main/images_/",
              xml_files_path="D:/Work/data/Main/images_/",
              labels_save_path="D:/Work/data/Main/labels_/")
obj.convert_xml_txt()
