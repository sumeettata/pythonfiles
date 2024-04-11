import argparse
import glob
from tqdm import tqdm
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--form', type=str, required=True)
args = parser.parse_args() 

for file in glob.glob(args.path+"//*//*//*//**")+glob.glob(args.path+"//*//*//**")+glob.glob(args.path+"//*//**")+glob.glob(args.path+"//**"):
    if file.contains(args.form):
        print(file)