from models.experimental import attempt_load
from utils.general import check_img_size

model = attempt_load('D://OneDrive//OneDrive - Tata Insights and Quants//Work//TATA Communications//Python files//downloads//stage_1_1_5k_06_02_2023_large.pt') 
stride = int(model.stride.max())
imgsz = check_img_size(640, s=stride)
names = model.module.names if hasattr(model, 'module') else model.names
print(names)