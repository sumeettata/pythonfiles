import cv2
import numpy as np
import glob
from tqdm import tqdm
import skimage.filters

def image_enhancement(image):
    contours,hierarchy = cv2.findContours(image, 2, 2)
    for w in contours:
        x,y,w,h = cv2.boundingRect(w)
        ar = image[y:y+h, x:x+w]
        white_pixel = np.count_nonzero(ar > 250)
        #print("x: "+str(x)+" y: "+str(y)+" w: "+str(w)+" h: "+str(h)+" area:"+str(w*h) + " white:"+str(white_pixel))
        if(white_pixel<200 and (x<5 or y<5 or x+w+5>image.shape[1] or y+h+5>image.shape[0])):
            for p in range(0,w):
                for q in range(0,h):
                    image[y+q, x+p] = 0
    if(image.shape[0]>image.shape[1]):
        side_add = np.zeros((image.shape[0], int((image.shape[0]-image.shape[1])/2)+5), dtype = "uint8")
        img_side = np.concatenate((side_add, image, side_add), axis=1)
        top_add = np.zeros((5, img_side.shape[1]), dtype = "uint8")
        img_vertical = np.concatenate((top_add, img_side, top_add), axis=0)
        final_image =cv2.resize(img_vertical,(img_vertical.shape[0],img_vertical.shape[0]),interpolation = cv2.INTER_AREA)
    elif(image.shape[0]<image.shape[1]):
        top_add = np.zeros((int((image.shape[1]-image.shape[0])/2)+5, image.shape[1]), dtype = "uint8")
        img_vertical = np.concatenate((top_add, image, top_add), axis=0)
        side_add = np.zeros((img_vertical.shape[0], 5), dtype = "uint8")
        img_side = np.concatenate((side_add, img_vertical, side_add), axis=1)
        final_image = cv2.resize(img_side,(img_vertical.shape[0],img_vertical.shape[0]),interpolation = cv2.INTER_AREA)
    else:
        side_add = np.zeros((image.shape[0], 5), dtype = "uint8")
        img_side = np.concatenate((side_add, image, side_add), axis=1)
        top_add = np.zeros((5, img_side.shape[1]), dtype = "uint8")
        img_vertical = np.concatenate((top_add, img_side, top_add), axis=0)
        final_image = cv2.resize(img_vertical,(img_vertical.shape[0],img_vertical.shape[0]),interpolation = cv2.INTER_AREA)
    return(final_image)

for file in tqdm(glob.glob('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//Vehicle orentation//front//*')):
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)
    t = skimage.filters.threshold_otsu(img_gray)
    res, img_thresh=cv2.threshold(img_gray,t,255,cv2.THRESH_BINARY)
    img2 = image_enhancement(img_thresh)
    print(img2.shape)
    # img3 = np.concatenate((img_gray,img2),axis=1)
    cv2.imshow("image",img2)
    key = cv2.waitKey(0)
    if key == 27: 
        break
cv2.destroyAllWindows()
