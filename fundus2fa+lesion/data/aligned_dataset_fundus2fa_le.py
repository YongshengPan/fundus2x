import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
# from data.base_dataset_paug import BaseDataset,  get_transform, normalize #get_params,
from PIL import Image
import pandas as pd
import cv2
import torchvision.transforms as transforms
import random
import numpy as np

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        df = opt.df.sample(frac=1).reset_index(drop=True)
        self.df = df
        self.w=opt.w
        self.B=opt.B
        self.A=opt.A
        print('*************************',df.shape,'*************************')
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.df.loc[index,self.A]   
        mer = cv2.imread(A_path)
        H,W=mer.shape[:2]
        A = mer[:,:W//2]
        B=mer[:,W//2:]
        B_tensor = inst_tensor = feat_tensor = 0

        h,w=A.shape[:2]
        if not self.opt.no_flip:
            # if w>700:
                aug=random.choice([0,0,0,0,1,2,3,4])
                if aug:
                    if aug<=3:
                        # delt=random.randint(w//10,w//3)
                        x = random.randint(0,w//3)
                        x2=  random.randint(x+w//3,w)
                        y = random.randint(0, h//3)
                        y2 = random.randint(y+h//3, h)

                        A = A[y:y2,x:x2,:]
                        B = B[y:y2,x:x2,:]
                    else:
                        t=random.randint(10,h//3+10)
                        b=random.randint(10,h//3+10)
                        l=random.randint(20,w//3+20)
                        r=random.randint(20,w//3+20)
                        A = cv2.copyMakeBorder(A,t,b,l,r,cv2.BORDER_CONSTANT,value=0)
                        B = cv2.copyMakeBorder(B,t,b,l,r,cv2.BORDER_CONSTANT,value=0)
        try:
            A = cv2.cvtColor(A,cv2.COLOR_BGR2RGB) 
        except:
            A = mer[:,:W//2]
            B=mer[:,W//2:]
        w=h=self.w
        A = cv2.resize(A,(w,h))
        B = cv2.resize(B,(w,h))

        params = get_params(self.opt, A.shape[:2])
        transform_B = get_transform(self.opt, params)    
        B_tensor = transform_B(Image.fromarray(B))
        A = Image.fromarray(A)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A)
        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path,}

        return input_dict

    def __len__(self):
        return len(self.df) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'