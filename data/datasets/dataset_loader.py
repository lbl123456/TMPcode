# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import re
import random
import os

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

class Nstr:
    def __init__(self, arg):
       self.x=arg
    def __sub__(self,other):
        c=self.x.replace(other.x,"")
        return c


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        Fake_img_path = self.get_FakeimgPath(img_path)
        Fake_img_1 = read_image(Fake_img_path)
        Fake_img_path = self.get_FakeimgPath(img_path)
        Fake_img_2 = read_image(Fake_img_path)
        if self.transform is not None:
            img = self.transform(img,Fake_img_1,Fake_img_2)

        return img, pid, camid, img_path

    def get_FakeimgPath(self,img_path):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid, camid = map(int, pattern.search(img_path).groups())
        Fake_camid = random.randint(1,6)
        if camid == Fake_camid:
            return img_path
        
        #获取 v3 文件夹路径
        filename = os.path.split(img_path)[1]
        v1 = Nstr(img_path)
        v2 = Nstr('bounding_box_train'+'\\'+filename)
        v3 = v1 - v2

        #filename - .jpg
        v5 = Nstr(filename)
        v6 = Nstr('.jpg')
        v7 = v5-v6

        v4 = v3 + 'bounding_box_train_camstyle' + '\\' + v7 + '_fake_' + str(camid) + 'to' + str(Fake_camid) + '.jpg'

        return v4
