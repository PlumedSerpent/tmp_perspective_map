import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import R2U_Net,AttU_Net,R2AttU_Net,Structure_U_Net,Analyzer_U_Net
import csv
from tqdm import tqdm as tqdm
from PIL import Image
import cv2
from progress.bar import Bar
import math
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class Solver(object):
    def __init__(self, config, train_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch

        # Hyper-parameters
        self.lr = config.lr
        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type =='U_Net':
            self.unet = U_Net(img_ch=3,output_ch=1)
        elif self.model_type =='R2U_Net':
            self.unet = R2U_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='AttU_Net':
            self.unet = AttU_Net(img_ch=3,output_ch=1)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3,output_ch=1,t=self.t)
        elif self.model_type =='ASM_U_Net':
            self.unet =Structure_U_Net(img_ch=3,output_ch=1)
            self.analyzer =Analyzer_U_Net(img_ch=1,output_ch=1)
            

        self.optimizer_unet = optim.Adam(params=list(self.unet.parameters()),
                                     lr= self.lr,weight_decay=1e-5)
        self.optimizer_analyzer = optim.Adam(params=list(self.analyzer.parameters()),
                                      lr=self.lr*0.2,weight_decay=1e-5)
        self.unet.to(self.device)
        self.unet = torch.nn.DataParallel(self.unet)
        self.analyzer.to(self.device)
        self.analyzer = torch.nn.DataParallel(self.analyzer)

        # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer_unet.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_analyzer.param_groups:
            param_group['lr'] = lr
    def reset_unet_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()
    def reset_analyzer_grad(self):
        """Zero the gradient buffers."""
        self.analyzer.zero_grad()
    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img
    def regular_loss(self,pred,gt):
        b,c,h,w=pred.size()
        row_std=torch.std(pred,dim=3)
        row_mean=torch.mean(pred,dim=3)
        thres_upper=(row_mean+row_std*2).view(b,c,h,1).expand_as(gt)
        thres_lower=(row_mean-row_std*2).view(b,c,h,1).expand_as(gt)
        idx_upper=torch.gt(pred,thres_upper)
        idx_lower=torch.gt(thres_lower,pred)
        loss=torch.zeros_like(pred)
        loss[idx_upper]=torch.pow((pred-thres_upper),2)[idx_upper]
        loss[idx_lower]=torch.pow((thres_lower-pred),2)[idx_lower]
        return loss.mean()

    def train(self):
        """Train encoder, generator and discriminator."""

        #====================================== Training ===========================================#
        #===========================================================================================#
        
        unet_path = os.path.join(self.model_path, 'S_%s-%d.pth' %(self.model_type,self.num_epochs))
        analyzer_path = os.path.join(self.model_path, 'A_%s-%d.pth' %(self.model_type,self.num_epochs))
        # U-Net Train

        # Train for Encoder
        lr = self.lr
        best_mse=1000.0
        start_epoch=0
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            self.analyzer.load_state_dict(torch.load(analyzer_path))
            print("Load from pretrained model!")
            start_epoch=42
        for epoch in range(start_epoch,self.num_epochs):
            w=math.pow(10,-epoch/self.num_epochs)

            self.unet.train(True)
            self.analyzer.train(True)
            epoch_loss = 0
            epoch_asm_loss = 0
            length = 0
            length_asm=0
            structure_losses = AverageMeter()
            bar = Bar('Training', max=len(self.train_loader))
            for batch_idx, (data) in enumerate(self.train_loader):
                # GT : Ground Truth
                
                image = data['image'].to(self.device)
                pmap = data['densitymap'].to(self.device)

                for step_idx in range(4):
                    self.reset_unet_grad()
                    structure_loss=0.
                    # SR : Segmentation Result
                    pmap_pred = self.unet(image)
                    pmap_feat,pmap_rec=self.analyzer(pmap)
                    pmap_pred_feat,pmap_pred_rec=self.analyzer(pmap_pred)
                    for feat_idx in range(len(pmap_pred_feat)):
                        structure_loss +=F.mse_loss(pmap_pred_feat[feat_idx],pmap_feat[feat_idx])
                    structure_loss/=len(pmap_pred_feat)
                    #structure_loss+=F.mse_loss(pmap_pred,pmap)
                    epoch_loss += structure_loss.item()
                    structure_losses.update(structure_loss.item(), image.size(0))
                    # Backprop + optimize
                    structure_loss.backward()
                    length += image.size(0)
                    self.optimizer_unet.step()
                bar.suffix  = '({batch}/{size})  | Total: {total:} | ETA: {eta:} | S Loss: {loss:.6f} |'.format(
                                batch=batch_idx + 1,
                                size=len(self.train_loader),
                                total=bar.elapsed_td,
                                eta=bar.eta_td,
                                loss=structure_losses.avg
                                )
                bar.next()
                
                self.reset_analyzer_grad()
                analyzer_loss=0.
                pmap_pred = self.unet(image)
                pmap_feat,pmap_rec=self.analyzer(pmap)
                pmap_pred_feat,pmap_pred_rec=self.analyzer(pmap_pred)
                for feat_idx in range(len(pmap_pred_feat)):
                    analyzer_loss +=((-0.2)*F.mse_loss(pmap_pred_feat[feat_idx],pmap_feat[feat_idx]))
                analyzer_loss/=len(pmap_pred_feat)
                analyzer_loss+=10*(F.mse_loss(pmap_rec,pmap))
                epoch_asm_loss += analyzer_loss.item()
                analyzer_loss.backward()
                self.optimizer_analyzer.step()
                length_asm += image.size(0)
            epoch_loss/=length
            epoch_asm_loss/=length_asm
            bar.finish()
            # Print the log info
            print('Epoch [%d/%d], Loss: %.6f, ASM Loss: %.6f' % (epoch+1, self.num_epochs, epoch_loss,epoch_asm_loss))
            
        

            # Decay learning rate
            if (epoch+1) in self.num_epochs_decay:
                for param_group in self.optimizer_analyzer.param_groups:
                    param_group['lr'] = self.lr*0.02
                for param_group in self.optimizer_unet.param_groups:
                    param_group['lr'] = self.lr*0.1
                self.lr=self.lr*0.1
                print ('Decay learning rate to lr: {}.'.format(self.lr))

            
            
            #===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()
            length=0
            epoch_loss = 0.
            for i, (data) in tqdm(enumerate(self.test_loader)):

                image = data['image'].to(self.device)
                pmap = data['densitymap'].to(self.device)

                # SR : Segmentation Result
                pmap_pred = self.unet(image)
                loss =F.mse_loss(pmap_pred,pmap)
                epoch_loss+=loss.item()
                pmap_pred=pmap_pred.detach().cpu().numpy()[0][0]
                # pmap_pred[pmap_pred>1]=1
                # pmap_pred[pmap_pred<0]=0
                pmap_pred=np.asarray(pmap_pred*255, dtype=np.uint8)


                feature_img = cv2.applyColorMap(pmap_pred, cv2.COLORMAP_JET)
                cv2.imwrite("result/%s/%d.png"%(self.model_type,i), feature_img)

                    
                length += image.size(0)
            epoch_loss/=length
            
            print('[Validation] Loss: %.6f'%(epoch_loss))
            
            '''
            torchvision.utils.save_image(images.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(SR.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
            torchvision.utils.save_image(GT.data.cpu(),
                                        os.path.join(self.result_path,
                                                    '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
            '''


            # Save Best U-Net model
            if epoch_loss < best_mse:
                best_mse = epoch_loss
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                best_analyzer=self.analyzer.state_dict()
                print('Best %s model score : %.4f'%(self.model_type,best_mse))
                torch.save(best_unet,unet_path)
                torch.save(best_analyzer,analyzer_path)
    
        

        
