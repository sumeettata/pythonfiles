import pandas as pd
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage_new/Road_imgs/sidewalk_1.v7i.png-mask-semantic/train/image_0006_png.rf.733d3e7e9f0440f7f60a2abd35c4c17e_mask.png',cv2.IMREAD_UNCHANGED)

print(img.shape)
plt.imshow(img)
plt.show()