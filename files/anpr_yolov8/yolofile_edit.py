import torch


model = torch.load('D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/Crowd_detection/train5/weights/epoch450.pt',map_location='cpu')
print(model['ema'].names)
model['ema'].names = {0:'head',1:'person'} 
model['model'].names = {0:'head',1:'person'} 
print(model['ema'].names)
torch.save(model,'D:/OneDrive/OneDrive - Tata Insights and Quants/Work/TATA Communications/GCP/Crowd_detection/train5/weights/crowd_450.pt')
