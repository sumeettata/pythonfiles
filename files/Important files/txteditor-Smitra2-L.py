import glob
import fileinput
from tqdm import tqdm
import os

path = 'C://Users//SumeetMitra//Downloads//vehicle-orientation-5'


files = glob.glob(path+'//*.txt')
os.mkdir(path+'_revised')

# dic = {'0':'0','1':'1','2':'2','3':'0','4':'1','5':'2','6':'0','7':'1','8':'2','9':'0','10':'1','11':'2'}
dic = {'0':'0','1':'1','2':'2','3':'3','4':'4','5':'5'}

for file in tqdm(files):
    l = open(file, "r")
    lst = []
    for line in l.readlines():
        m = line.rstrip()
        m = m.split()
        if m[0] in dic.keys():
            m[0] = dic[m[0]]
            n = ' '.join(m)
            lst.append(n)
    with open(path+'_revised//'+os.path.basename(file), 'w') as f:
        f.write('\n'.join(lst))
        f.close()
    l.close()