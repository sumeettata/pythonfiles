import os
import os
import matplotlib.pyplot as plt
import cv2

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/garbage/Mask_19.png'

img = cv2.imread(path,cv2.IMREAD_UNCHANGED)

plt.imshow(img)
plt.show()