from ultralytics.yolo.v8.detect.train import DetectionTrainer
from ultralytics.yolo.v8.segment.train import SegmentationTrainer

import yaml
import os

##git
# change current directory
current_dir = os.getcwd()
os.chdir('/CV/model_training/ultralytics_v2/')

# load the Training module
with open("Training.yaml", "r") as stream:
    dict_info = yaml.safe_load(stream)

print(dict_info)

# create path
if not os.path.exists('Results'):
    os.mkdir('Results')

if dict_info['xml_to_txt']:
    from utils.xml2txt import xml2txt
    xml2txt(dict_info['path'],dict_info['names'])

if dict_info['json_to_txt']:
    from utils.json2txt import json2txt
    json2txt(dict_info['path'],dict_info['names'])

# create data.yaml
with open("Results/data.yaml", "w") as stream:
    yaml.dump({'path': dict_info['path'], 'train': dict_info['path'] + '/train.txt', 'val': dict_info['path'] + '/val.txt', 'test': '', 'names': dict_info['names'] },stream, default_flow_style=False)

# create the over write cfg
args = { 
    key: dict_info[key] for key in ['model','epochs','save_period','optimizer','resume','lr0']
}

args['data'] = "Results/data.yaml"
args['device'] = None
args['project'] = 'Results/'+dict_info['project']



#start the 
if dict_info['mode'] == 'detect':
    trainer = DetectionTrainer(overrides=args)
    trainer.train()

elif dict_info['mode'] == 'segment':
    trainer = SegmentationTrainer(overrides=args)
    trainer.train()

