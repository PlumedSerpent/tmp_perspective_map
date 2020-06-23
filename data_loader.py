#%%
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F_vision
import torch.nn.functional as F
import pandas as pd
import scipy.io as scio
class CrowdDataset(torch.utils.data.Dataset):
    '''
    CrowdDataset
    '''

    def __init__(self, root, phase, main_transform=None, img_transform=None, dmap_transform=None):
        '''
        root: the root path of dataset.
        phase: train or test.
        main_transform: transforms on both image and density map.
        img_transform: transforms on image.
        dmap_transform: transforms on densitymap.
        '''
        self.img_path = os.path.join(root, phase+'_frame/')
        self.dmap_path = os.path.join(root, phase+'_perspective/')
        self.data_files = [filename for filename in os.listdir(self.img_path)
                           if os.path.isfile(os.path.join(self.img_path, filename))]
    
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.dmap_transform = dmap_transform
        self.phase=phase

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        index = index % len(self.data_files)
        fname = self.data_files[index]
        img, dmap = self.read_image_and_dmap(fname)

        if self.main_transform is not None:
            img, dmap = self.main_transform((img, dmap))
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.dmap_transform is not None:
            dmap = self.dmap_transform(dmap)

        # if self.phase=="test":
        #     pmap_pred=dmap.numpy()[0]
        #         # pmap_pred[pmap_pred>1]=1
        #         # pmap_pred[pmap_pred<0]=0
        #     pmap_pred=np.asarray(pmap_pred*255, dtype=np.uint8)


        #     feature_img = cv2.applyColorMap(pmap_pred, cv2.COLORMAP_JET)
        #     cv2.imwrite("GT/%d.png"%(index), feature_img)

        return {'image': img, 'densitymap': dmap}

    def read_image_and_dmap(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            # print('There is a grayscale image.')
            img = img.convert('RGB')
        image_id=fname[:6]
        dmap =  scio.loadmat(os.path.join(
            self.dmap_path, image_id + '.mat'))
        dmap = dmap['pMap'].astype(np.float32, copy=False)
        dmap=dmap/255.0
        dmap = Image.fromarray(dmap)

        return img, dmap
def create_train_dataloader(root, use_flip, batch_size):
    '''
    Create train dataloader.
    root: the dataset root.
    use_flip: True or false.
    batch size: the batch size.
    '''
    main_trans_list = []
    if use_flip:
        main_trans_list.append(RandomHorizontalFlip())
    main_trans_list.append(PairedCrop())
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.5,0.5,0.5],std=[0.225,0.225,0.225])])
    dmap_trans = ToTensor()
    dataset = CrowdDataset(root=root, phase='train', main_transform=main_trans, 
                    img_transform=img_trans,dmap_transform=dmap_trans)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=12)
    return dataloader

def create_test_dataloader(root):
    '''
    Create train dataloader.
    root: the dataset root.
    '''
    main_trans_list = []
    main_trans_list.append(PairedCrop())
    main_trans = Compose(main_trans_list)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.5,0.5,0.5],std=[0.225,0.225,0.225])])
    dmap_trans = ToTensor()
    dataset = CrowdDataset(root=root, phase='test', main_transform=main_trans, 
                    img_transform=img_trans,dmap_transform=dmap_trans)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    return dataloader

#----------------------------------#
#          Transform code          #
#----------------------------------#
class RandomHorizontalFlip(object):
    '''
    Random horizontal flip.
    prob = 0.5
    '''
    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = img_and_dmap
        if random.random() < 0.5:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), dmap.transpose(Image.FLIP_LEFT_RIGHT))
        else:
            return (img, dmap)

class PairedCrop(object):
    '''
    Paired Crop for both image and its density map.
    Note that due to the maxpooling in the nerual network, 
    we must promise that the size of input image is the corresponding factor.
    '''
    def __init__(self, factor=16):
        self.factor = factor

    @staticmethod
    def get_params(img, factor):
        w, h = img.size
        if w % factor == 0 and h % factor == 0:
            return 0, 0, h, w
        else:
            return 0, 0, h - (h % factor), w - (w % factor)

    def __call__(self, img_and_dmap):
        '''
        img: PIL.Image
        dmap: PIL.Image
        '''
        img, dmap = img_and_dmap
        
        i, j, th, tw = self.get_params(img, self.factor)

        img = F_vision.crop(img, i, j, th, tw)
        dmap = F_vision.crop(dmap, i, j, th, tw)
        img =img.resize((256, 256),Image.ANTIALIAS)
        dmap =dmap.resize((256, 256),Image.ANTIALIAS)
        return (img, dmap)
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

# testing code
#[6.60696751e-05 1.53686842e-04 3.03317757e-04 5.70904907e-04 1.09181233e-03 2.36344733e-03 4.39890302e-03 7.51745701e-03]
#[54             ,93,75,25,56,22]
if __name__ == "__main__":
    root = '/home/wangjian/CrowdCounting/WorldExpo/'
    #dataloader = create_train_dataloader(root,False,1)
    dataloader = create_test_dataloader(root)
    vector_np=[]
    for i, data in enumerate(dataloader):
        image = data['image']
        densitymap = data['densitymap']



        



        
            #print(image.shape,densitymap.shape)
   
    