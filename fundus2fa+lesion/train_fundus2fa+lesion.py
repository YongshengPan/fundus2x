from macpath import split
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
# from subprocess import call
import fractions
import cv2
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions

from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from datetime import datetime
# from data.versions.korniaaugmentation import trainpAugmentation
import kornia
from skimage.util import img_as_ubyte
import pandas as pd
opt = TrainOptions().parse()
from options.test_options import TestOptions

from sklearn.metrics import confusion_matrix,roc_curve, auc, roc_auc_score,f1_score
from show import splitdf

opt.resize_or_crop = ''
# opt.resize_or_crop='scale_width_crop'
opt.save_epoch_freq=1
opt.display_freq = 500
# opt.display_freq = 10
opt.print_freq = opt.display_freq
opt.batchSize = 2
# opt.niter=5
opt.niter=10
opt.niter_decay=10 #两个加起来

opt.name = 'fundus2fa_le'
opt.seg_loss = 1
opt.load_pretrain='/home/healgoo/risk_factor/vessel/generation/super/pix2pixHD/checkpoints/fundus2fa_le/03-12-1939_4.4_6min_512'
df=pd.read_csv('/home/healgoo/data/zaoying/warped/fundus2fa/co2faseries_crop.csv')
df['time'] = df['SeriesDescription'].str.extract(r'(\d+:\d+\.\d+)')
df['time'] = pd.to_datetime(df['time'],format= '%M:%S.%f',errors='coerce')
df.loc[(df.time>=pd.to_datetime('1900-01-01 00:00:40.000'))&(df.time<pd.to_datetime('1900-01-01 00:01:30.000')),'mark']='venous'
df.loc[(df.time>=pd.to_datetime('1900-01-01 00:04:00.000'))&(df.time<pd.to_datetime('1900-01-01 00:06:00.000')),'mark']='late'

df=df[df.mark=='late']
other = df[df.ischemia_n<=1].sample(500,random_state=2023)
df=df[df.ischemia_n>1].append(other)
# trange='40_60s'
trange='4.4_6min'
print(df.shape,'--------------------------------------------------')
opt.A='crop'
opt.B=opt.A
splitid='orgid'
TRAIN,VALID=splitdf(df,by=splitid,test_size=.2)
VALID,TEST=splitdf(VALID,by=splitid,test_size=.5)
opt.w=512
# opt.w=768
import glob
ls=glob.glob(opt.load_pretrain+'/*G.pth')
ls=[x.split('/')[-1].split('_')[0] for x in ls]
print(ls)
if len(ls)>0:
    opt.which_epoch=str(ls[-1])

now = datetime.now()
opt.df=TRAIN

opttest = TestOptions().parse(save=False)
opttest.df=VALID
opttest.nThreads = 1   # test code only supports nThreads = 1
opttest.serial_batches = True  # no shuffle
opttest.no_flip = True  # no flip
opttest.name = opt.name
opttest.no_flip = True 
opttest.name = opt.name
opttest.w=opt.w
opttest.A=opt.A
opttest.B=opt.B
data_loader = CreateDataLoader(opt)
test_loader = CreateDataLoader(opttest)
opt.name = opt.name+'/'+now.strftime("%m-%d-%H%M")+'_'+trange+'_'+str(opt.w)
visualizer = Visualizer(opt)
TEST.to_csv(os.path.join(opt.checkpoints_dir, opt.name, 'test.csv'))
VALID.to_csv(os.path.join(opt.checkpoints_dir, opt.name, 'valid.csv'))
TRAIN.to_csv(os.path.join(opt.checkpoints_dir, opt.name, 'train.csv'))
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

traindataset = data_loader.load_data()
test_dataset = test_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)

optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

total_steps = (start_epoch-1) * dataset_size + epoch_iter

display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

best_auc=.5
best_f1=.4
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(traindataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################

        losses, generated = model(Variable(data['label']), Variable(data['inst']), 
            Variable(data['image']), Variable(data['feat']), infer=save_fake)

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        # 
        if opt.seg_loss:
            loss_G = loss_dict['G_GAN']*2 + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) +loss_dict['G_seg']
        else:
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()          
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()        
        optimizer_D.step()        

        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}    
            errors.update({'loss_D':loss_D,'loss_G':loss_G})           
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
        ### display output images
        if save_fake:
            lb = util.tensor2label(data['label'][0], opt.label_nc)
            real = util.tensor2im(data['image'][0])
            gen = util.tensor2im(generated.data[0])
            mer = np.hstack([lb,real,gen])
            visuals = OrderedDict([('vis', mer)])
            visualizer.display_current_results(visuals, epoch, total_steps)
            
        if epoch_iter >= dataset_size:
            break

    t=time.time() - epoch_start_time
    print('End of epoch %d / %d \t Time Taken: %d sec' %
        (epoch, opt.niter + opt.niter_decay, t))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
        
    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()
