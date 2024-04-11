import glob
import os

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/Garbage/Garbage_dataset_roboflow/'

for file in glob.glob(path+'*/labels/*'):
    lst1 = []
    with open(file) as f:
        for line in f.readlines():
            lst = line.split(' ')
            print(line)
            lst[0] = '0'
            line = ' '.join(lst)
            #line = ' '.join(line)
            lst1.append(line)
        print(lst1)
    f.close()
    if not os.path.exists(os.path.dirname(file)+'_new'):
        os.mkdir(os.path.dirname(file)+'_new')
    with open(os.path.dirname(file)+'_new/'+os.path.basename(file),'w') as f:
        for lin in lst1:
            f.write(lin)
    f.close()
                