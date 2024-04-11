import cv2
import pandas as pd 
import os

path = 'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/Data_1/ANPR Stage2/testset_data_final/Results.csv'
df = pd.read_csv(path)


lst_pred=[]
lst_truth=[]
i = 0
j = 0
for index,row in df.iterrows():
    if len(str(row['predicted'])) == len(str(row['plate_number'])):
        pred = [x for x in str(row['predicted'])]
        truth = [x for x in str(row['plate_number'])]
        lst_pred.extend(pred)
        lst_truth.extend(truth)
        i = i + 1
    j = j + 1
    
label = [chr(v).upper() for v in range(97, 123)]+[str(x) for x in range(10)]

from sklearn.metrics import confusion_matrix

results = confusion_matrix(lst_truth, lst_pred,labels=label)   
df_cm = pd.DataFrame(results, index=label, columns=label) 
df_cm.to_csv(os.path.dirname(path)+'/confusionmatrix.csv')