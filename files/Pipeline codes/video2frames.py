import cv2
import os 

videos_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR//03-03'
save_path = 'D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Data_1//ANPR//03-03_mov'


dir_path = videos_path
root_path = save_path
dir_list = os.listdir(dir_path)
files = [x for x in dir_list if x.endswith(".mp4")]

def makeimage(video_name):
    video_name2 = os.path.join(dir_path,str(video_name))
    cap = cv2.VideoCapture(video_name2)
    loc_path = str(video_name)
    path = os.path.join(root_path, loc_path)
    os.mkdir(path.split('.')[0])
    i = 0
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == 0:
            print('Number of Files : ' + str(i))
            break
        else:
            print('creating ' + str(i), end="\r")
            count+= 16
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            cv2.imwrite(path.split('.')[0]+ '\\' + loc_path.split('.')[0] +'_'+str(i+1)+'.jpg', frame)
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    
j = 0
for file in files:
    print('Video folder : ' + str(j+1))
    makeimage(file)
    j += 1