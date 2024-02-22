import os
import shutil
from tqdm import tqdm

source = 'D:/dataset/coco2017/train2017'
destination = './datasets/text2style/trainA'

for file in tqdm(os.listdir(source)):
    if file.endswith('.jpg'):
        shutil.copy(os.path.join(source, file), os.path.join(destination, file))