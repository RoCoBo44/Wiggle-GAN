import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.cuda as cu
import torch.optim as optim
import pickle
from torchvision import transforms
from utils import augmentData, RGBtoL, LtoRGB
from PIL import Image
from dataloader import dataloader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from datetime import date
from statistics import mean
from architectures import depth_generator_UNet, \
    depth_discriminator_noclass_UNet

class WiggleGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.nCameras = args.cameras
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.class_num = (args.cameras - 1) * 2  # un calculo que hice en paint
        self.sample_num = self.class_num ** 2
        self.imageDim = args.imageDim
        self.epochVentaja = args.epochV
        self.cantImages = args.cIm
        self.visdom = args.visdom
        self.lambdaL1 = args.lambdaL1
        self.depth = args.depth

        self.clipping = args.clipping
        self.WGAN = False
        if (self.clipping > 0):
            self.WGAN = True

        self.seed = str(random.randint(0, 99999))
        self.seed_load = args.seedLoad
        self.toLoad = False
        if (self.seed_load != "-0000"):
            self.toLoad = True

        self.zGenFactor = args.zGF
        self.zDisFactor = args.zDF
        self.bFactor = args.bF
        self.CR = False
        if (self.zGenFactor > 0 or self.zDisFactor > 0 or self.bFactor > 0):
            self.CR = True

        self.expandGen = args.expandGen
        self.expandDis = args.expandDis

        self.wiggleDepth = args.wiggleDepth
        self.wiggle = False
        if (self.wiggleDepth > 0):
            self.wiggle = True



        # load dataset

        self.onlyGen = args.lrD <= 0 

        if not self.wiggle:
            self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim, split='train',
                                      trans=not self.CR)

            self.data_Validation = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim,
                                          split='validation')

            self.dataprint = self.data_Validation.__iter__().__next__()

            data = self.data_loader.__iter__().__next__().get('x_im')


            if not self.onlyGen:
              self.D = depth_discriminator_noclass_UNet(input_dim=3, output_dim=1, input_shape=data.shape,
                                                        class_num=self.class_num,
                                                        expand_net=self.expandDis, depth = self.depth, wgan = self.WGAN)
              self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        self.data_Test = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim, split='test')
        self.dataprint_test = self.data_Test.__iter__().__next__()

        # networks init

        self.G = depth_generator_UNet(input_dim=4, output_dim=3, class_num=self.class_num, expand_net=self.expandGen, depth = self.depth)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))


        if self.gpu_mode:
            self.G.cuda()
            if not self.wiggle and not self.onlyGen:
                self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.L1 = nn.L1Loss().cuda()
            self.MSE = nn.MSELoss().cuda()
            self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE = nn.MSELoss()
            self.L1 = nn.L1Loss()
            self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        if not self.wiggle and not self.onlyGen:
            utils.print_network(self.D)
        print('-----------------------------------------------')

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i * self.class_num: (i + 1) * self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
             self.sample_y_ = self.sample_y_.cuda()

        if (self.toLoad):
            self.load()

    def train(self):

        if self.visdom:
            random.seed(time.time())
            today = date.today()

            vis = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
            visValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
            visEpoch = utils.VisdomLineTwoPlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
            visImages = utils.VisdomImagePlotter(env_name='Cobo_depth_Images_' + str(today) + '_' + self.seed)
            visImagesTest = utils.VisdomImagePlotter(env_name='Cobo_depth_ImagesTest_' + str(today) + '_' + self.seed)

            visLossGTest = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
            visLossGValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)

            visLossDTest = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
            visLossDValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)

        self.train_hist = {}
        self.epoch_hist = {}
        self.details_hist = {}
        self.train_hist['D_loss_train'] = []
        self.train_hist['G_loss_train'] = []
        self.train_hist['D_loss_Validation'] = []
        self.train_hist['G_loss_Validation'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.details_hist['G_T_Comp_im'] = []
        self.details_hist['G_T_BCE_fake_real'] = []
        self.details_hist['G_T_Cycle'] = []
        self.details_hist['G_zCR'] = []

        self.details_hist['G_V_Comp_im'] = []
        self.details_hist['G_V_BCE_fake_real'] = []
        self.details_hist['G_V_Cycle'] = []

        self.details_hist['D_T_BCE_fake_real_R'] = []
        self.details_hist['D_T_BCE_fake_real_F'] = []
        self.details_hist['D_zCR'] = []
        self.details_hist['D_bCR'] = []

        self.details_hist['D_V_BCE_fake_real_R'] = []
        self.details_hist['D_V_BCE_fake_real_F'] = []

        self.epoch_hist['D_loss_train'] = []
        self.epoch_hist['G_loss_train'] = []
        self.epoch_hist['D_loss_Validation'] = []
        self.epoch_hist['G_loss_Validation'] = []

        ##Para poder tomar el promedio por epoch
        iterIniTrain = 0
        iterFinTrain = 0

        iterIniValidation = 0
        iterFinValidation = 0

        maxIter = self.data_loader.dataset.__len__() // self.batch_size
        maxIterVal = self.data_Validation.dataset.__len__() // self.batch_size

        if (self.WGAN):
            one = torch.tensor(1, dtype=torch.float).cuda()
            mone = one * -1
        else:
            self.y_real_ = torch.ones(self.batch_size, 1)
            self.y_fake_ = torch.zeros(self.batch_size, 1)
            if self.gpu_mode:
                self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        print('training start!!')
        start_time = time.time()

        for epoch in range(self.epoch):

            if (epoch < self.epochVentaja):
                ventaja = True
            else:
                ventaja = False

            self.G.train()

            if not self.onlyGen:
              self.D.train()
            epoch_start_time = time.time()


            # TRAIN!!!
            for iter, data in enumerate(self.data_loader):

                x_im = data.get('x_im')
                x_dep = data.get('x_dep')
                y_im = data.get('y_im')
                y_dep = data.get('y_dep')
                y_ = data.get('y_')

                # x_im  = imagenes normales
                # x_dep = profundidad de images
                # y_im  = imagen con el angulo cambiado
                # y_    = angulo de la imagen = tengo que tratar negativos

                # Aumento mi data
                if (self.CR):
                    x_im_aug, y_im_aug = augmentData(x_im, y_im)
                    x_im_vanilla = x_im

                    if self.gpu_mode:
                        x_im_aug, y_im_aug = x_im_aug.cuda(), y_im_aug.cuda()

                if iter >= maxIter:
                    break

                if self.gpu_mode:
                    x_im, y_, y_im, x_dep, y_dep = x_im.cuda(), y_.cuda(), y_im.cuda(), x_dep.cuda(), y_dep.cuda()

                # update D network
                if not ventaja and not self.onlyGen:

                    for p in self.D.parameters():  # reset requires_grad
                        p.requires_grad = True  # they are set to False below in netG update

                    self.D_optimizer.zero_grad()

                    # Real Images
                    D_real, D_features_real = self.D(y_im, x_im, y_dep, y_)  ## Es la funcion forward `` g(z) x

                    # Fake Images
                    G_, G_dep = self.G( y_, x_im, y_dep)
                    D_fake, D_features_fake = self.D(G_, x_im, G_dep, y_)

                    # Losses
                    #  GAN Loss
                    if (self.WGAN): # de WGAN
                        D_loss_real_fake_R = - torch.mean(D_real)
                        D_loss_real_fake_F = torch.mean(D_fake)
                        #D_loss_real_fake_R = - D_loss_real_fake_R_positive

                    else:       # de Gan normal
                        D_loss_real_fake_R = self.BCEWithLogitsLoss(D_real, self.y_real_)
                        D_loss_real_fake_F = self.BCEWithLogitsLoss(D_fake, self.y_fake_)

                    D_loss = D_loss_real_fake_F + D_loss_real_fake_R

                    if self.CR:

                        # Fake Augmented Images bCR
                        x_im_aug_bCR, G_aug_bCR = augmentData(x_im_vanilla, G_.data.cpu())

                        if self.gpu_mode:
                            G_aug_bCR, x_im_aug_bCR = G_aug_bCR.cuda(), x_im_aug_bCR.cuda()

                        D_fake_bCR, D_features_fake_bCR = self.D(G_aug_bCR, x_im_aug_bCR, G_dep, y_)
                        D_real_bCR, D_features_real_bCR = self.D(y_im_aug, x_im_aug, y_dep, y_)

                        # Fake Augmented Images zCR
                        G_aug_zCR, G_dep_aug_zCR = self.G(y_, x_im_aug, x_dep)
                        D_fake_aug_zCR, D_features_fake_aug_zCR = self.D(G_aug_zCR, x_im_aug, G_dep_aug_zCR, y_)

                        #  bCR Loss (*)
                        D_loss_real = self.MSE(D_features_real, D_features_real_bCR)
                        D_loss_fake = self.MSE(D_features_fake, D_features_fake_bCR)
                        D_bCR = (D_loss_real + D_loss_fake) * self.bFactor

                        #  zCR Loss
                        D_zCR = self.MSE(D_features_fake, D_features_fake_aug_zCR) * self.zDisFactor

                        D_CR_losses = D_bCR + D_zCR
                        #D_CR_losses.backward(retain_graph=True)

                        D_loss += D_CR_losses

                        self.details_hist['D_bCR'].append(D_bCR.detach().item())
                        self.details_hist['D_zCR'].append(D_zCR.detach().item())
                    else:
                        self.details_hist['D_bCR'].append(0)
                        self.details_hist['D_zCR'].append(0)

                    self.train_hist['D_loss_train'].append(D_loss.detach().item())
                    self.details_hist['D_T_BCE_fake_real_R'].append(D_loss_real_fake_R.detach().item())
                    self.details_hist['D_T_BCE_fake_real_F'].append(D_loss_real_fake_F.detach().item())
                    if self.visdom:
                      visLossDTest.plot('Discriminator_losses',
                                           ['D_T_BCE_fake_real_R','D_T_BCE_fake_real_F', 'D_bCR', 'D_zCR'], 'train',
                                           self.details_hist)
                    #if self.WGAN:
                    #    D_loss_real_fake_F.backward(retain_graph=True)
                    #    D_loss_real_fake_R_positive.backward(mone)
                    #else:
                    #    D_loss_real_fake.backward()
                    D_loss.backward()

                    self.D_optimizer.step()

                    #WGAN
                    if (self.WGAN):
                        for p in self.D.parameters():
                            p.data.clamp_(-self.clipping, self.clipping) #Segun paper si el valor es muy chico lleva al banishing gradient
                    # Si se aplicaria la mejora en las WGANs tendiramos que sacar los batch normalizations de la red


                # update G network
                self.G_optimizer.zero_grad()

                G_, G_dep = self.G(y_, x_im, x_dep)

                if not ventaja and not self.onlyGen:
                    for p in self.D.parameters():
                        p.requires_grad = False  # to avoid computation

                    # Fake images
                    D_fake, _ = self.D(G_, x_im, G_dep, y_)

                    if (self.WGAN):
                        G_loss_fake = -torch.mean(D_fake) #de WGAN
                    else:
                        G_loss_fake = self.BCEWithLogitsLoss(D_fake, self.y_real_)

                    # loss between images (*)
                    #G_join = torch.cat((G_, G_dep), 1)
                    #y_join = torch.cat((y_im, y_dep), 1)

                    G_loss_Comp = self.L1(G_, y_im) 
                    if self.depth:
                      G_loss_Comp += self.L1(G_dep, y_dep)

                    G_loss_Dif_Comp = G_loss_Comp * self.lambdaL1

                    reverse_y = - y_ + 1
                    reverse_G, reverse_G_dep = self.G(reverse_y, G_, G_dep)
                    G_loss_Cycle = self.L1(reverse_G, x_im) 
                    if self.depth:
                      G_loss_Cycle += self.L1(reverse_G_dep, x_dep) 
                    G_loss_Cycle = G_loss_Cycle * self.lambdaL1/2


                    if (self.CR):
                        # Fake images augmented

                        G_aug, G_dep_aug = self.G(y_, x_im_aug, x_dep)
                        D_fake_aug, _ = self.D(G_aug, x_im, G_dep_aug, y_)

                        if (self.WGAN):
                            G_loss_fake = - (torch.mean(D_fake)+torch.mean(D_fake_aug))/2
                        else:
                            G_loss_fake = ( self.BCEWithLogitsLoss(D_fake, self.y_real_) +
                                            self.BCEWithLogitsLoss(D_fake_aug,self.y_real_)) / 2

                        # loss between images (*)
                        #y_aug_join = torch.cat((y_im_aug, y_dep), 1)
                        #G_aug_join = torch.cat((G_aug, G_dep_aug), 1)

                        G_loss_Comp_Aug = self.L1(G_aug, y_im_aug)
                        if self.depth:
                           G_loss_Comp_Aug += self.L1(G_dep_aug, y_dep)
                        G_loss_Dif_Comp = (G_loss_Comp + G_loss_Comp_Aug)/2 * self.lambdaL1


                    G_loss = G_loss_fake + G_loss_Dif_Comp + G_loss_Cycle

                    self.details_hist['G_T_BCE_fake_real'].append(G_loss_fake.detach().item())
                    self.details_hist['G_T_Comp_im'].append(G_loss_Dif_Comp.detach().item())
                    self.details_hist['G_T_Cycle'].append(G_loss_Cycle.detach().item())
                    self.details_hist['G_zCR'].append(0)


                else:

                    G_loss = self.L1(G_, y_im) 
                    if self.depth:
                      G_loss += self.L1(G_dep, y_dep)
                    G_loss = G_loss * self.lambdaL1
                    self.details_hist['G_T_Comp_im'].append(G_loss.detach().item())
                    self.details_hist['G_T_BCE_fake_real'].append(0)
                    self.details_hist['G_T_Cycle'].append(0)
                    self.details_hist['G_zCR'].append(0)

                G_loss.backward()
                self.G_optimizer.step()
                self.train_hist['G_loss_train'].append(G_loss.detach().item())
                if self.onlyGen:
                  self.train_hist['D_loss_train'].append(0)

                iterFinTrain += 1

                if self.visdom:
                  visLossGTest.plot('Generator_losses',
                                      ['G_T_Comp_im', 'G_T_BCE_fake_real', 'G_zCR','G_T_Cycle'],
                                       'train', self.details_hist)

                  vis.plot('loss', ['D_loss_train', 'G_loss_train'], 'train', self.train_hist)

            ##################Validation####################################
            with torch.no_grad():

                self.G.eval()
                if not self.onlyGen:
                  self.D.eval()

                for iter, data in enumerate(self.data_Validation):

                    # Aumento mi data
                    x_im = data.get('x_im')
                    x_dep = data.get('x_dep')
                    y_im = data.get('y_im')
                    y_dep = data.get('y_dep')
                    y_ = data.get('y_')
                    # x_im  = imagenes normales
                    # x_dep = profundidad de images
                    # y_im  = imagen con el angulo cambiado
                    # y_    = angulo de la imagen = tengo que tratar negativos

                    # x_im  = torch.Tensor(list(x_im))
                    # x_dep = torch.Tensor(x_dep)
                    # y_im  = torch.Tensor(y_im)
                    # print(y_.shape[0])
                    if iter == maxIterVal:
                        # print ("Break")
                        break
                    # print (y_.type(torch.LongTensor).unsqueeze(1))


                    # print("y_vec_", y_vec_)
                    # print ("z_", z_)

                    if self.gpu_mode:
                        x_im, y_, y_im, x_dep, y_dep = x_im.cuda(), y_.cuda(), y_im.cuda(), x_dep.cuda(), y_dep.cuda()
                    # D network

                    if not ventaja and not self.onlyGen:
                        # Real Images
                        D_real, _ = self.D(y_im, x_im, y_dep,y_)  ## Es la funcion forward `` g(z) x

                        # Fake Images
                        G_, G_dep = self.G(y_, x_im, x_dep)
                        D_fake, _ = self.D(G_, x_im, G_dep, y_)
                        # Losses
                        #  GAN Loss
                        if (self.WGAN):  # de WGAN
                            D_loss_real_fake_R = - torch.mean(D_real)
                            D_loss_real_fake_F = torch.mean(D_fake)

                        else:  # de Gan normal
                            D_loss_real_fake_R = self.BCEWithLogitsLoss(D_real, self.y_real_)
                            D_loss_real_fake_F = self.BCEWithLogitsLoss(D_fake, self.y_fake_)

                        D_loss_real_fake = D_loss_real_fake_F + D_loss_real_fake_R

                        D_loss = D_loss_real_fake

                        self.train_hist['D_loss_Validation'].append(D_loss.item())
                        self.details_hist['D_V_BCE_fake_real_R'].append(D_loss_real_fake_R.item())
                        self.details_hist['D_V_BCE_fake_real_F'].append(D_loss_real_fake_F.item())
                        if self.visdom:
                          visLossDValidation.plot('Discriminator_losses',
                                                     ['D_V_BCE_fake_real_R','D_V_BCE_fake_real_F'], 'Validation',
                                                     self.details_hist)

                    # G network

                    G_, G_dep = self.G(y_, x_im, x_dep)

                    if not ventaja and not self.onlyGen:
                        # Fake images
                        D_fake,_ = self.D(G_, x_im, G_dep, y_)

                        #Loss GAN
                        if (self.WGAN):
                            G_loss = -torch.mean(D_fake)  # porWGAN
                        else:
                            G_loss = self.BCEWithLogitsLoss(D_fake, self.y_real_) #de GAN NORMAL

                        self.details_hist['G_V_BCE_fake_real'].append(G_loss.item())

                        #Loss comparation
                        #G_join = torch.cat((G_, G_dep), 1)
                        #y_join = torch.cat((y_im, y_dep), 1)

                        G_loss_Comp = self.L1(G_, y_im)
                        if self.depth:
                          G_loss_Comp += self.L1(G_dep, y_dep)
                        G_loss_Comp = G_loss_Comp * self.lambdaL1

                        reverse_y = - y_ + 1                  
                        reverse_G, reverse_G_dep = self.G(reverse_y, G_, G_dep)
                        G_loss_Cycle = self.L1(reverse_G, x_im) 
                        if self.depth:
                          G_loss_Cycle += self.L1(reverse_G_dep, x_dep) 
                        G_loss_Cycle = G_loss_Cycle * self.lambdaL1/2

                        G_loss += G_loss_Comp + G_loss_Cycle 


                        self.details_hist['G_V_Comp_im'].append(G_loss_Comp.item())
                        self.details_hist['G_V_Cycle'].append(G_loss_Cycle.detach().item())

                    else:
                        G_loss = self.L1(G_, y_im) 
                        if self.depth:
                          G_loss += self.L1(G_dep, y_dep)
                        G_loss = G_loss * self.lambdaL1
                        self.details_hist['G_V_Comp_im'].append(G_loss.item())
                        self.details_hist['G_V_BCE_fake_real'].append(0)
                        self.details_hist['G_V_Cycle'].append(0)

                    self.train_hist['G_loss_Validation'].append(G_loss.item())
                    if self.onlyGen:
                      self.train_hist['D_loss_Validation'].append(0)


                    iterFinValidation += 1
                    if self.visdom:
                      visLossGValidation.plot('Generator_losses', ['G_V_Comp_im', 'G_V_BCE_fake_real','G_V_Cycle'],
                                                 'Validation', self.details_hist)
                      visValidation.plot('loss', ['D_loss_Validation', 'G_loss_Validation'], 'Validation',
                                           self.train_hist)

            ##Vis por epoch

            if ventaja or self.onlyGen:
                self.epoch_hist['D_loss_train'].append(0)
                self.epoch_hist['D_loss_Validation'].append(0)
            else:
                #inicioTr = (epoch - self.epochVentaja) * (iterFinTrain - iterIniTrain)
                #inicioTe = (epoch - self.epochVentaja) * (iterFinValidation - iterIniValidation)
                self.epoch_hist['D_loss_train'].append(mean(self.train_hist['D_loss_train'][iterIniTrain: -1]))
                self.epoch_hist['D_loss_Validation'].append(mean(self.train_hist['D_loss_Validation'][iterIniValidation: -1]))

            self.epoch_hist['G_loss_train'].append(mean(self.train_hist['G_loss_train'][iterIniTrain:iterFinTrain]))
            self.epoch_hist['G_loss_Validation'].append(
                mean(self.train_hist['G_loss_Validation'][iterIniValidation:iterFinValidation]))
            if self.visdom:
              visEpoch.plot('epoch', epoch,
                               ['D_loss_train', 'G_loss_train', 'D_loss_Validation', 'G_loss_Validation'],
                               self.epoch_hist)

            self.train_hist['D_loss_train'] = self.train_hist['D_loss_train'][-1:]
            self.train_hist['G_loss_train'] = self.train_hist['G_loss_train'][-1:]
            self.train_hist['D_loss_Validation'] = self.train_hist['D_loss_Validation'][-1:]
            self.train_hist['G_loss_Validation'] = self.train_hist['G_loss_Validation'][-1:]
            self.train_hist['per_epoch_time'] = self.train_hist['per_epoch_time'][-1:]
            self.train_hist['total_time'] = self.train_hist['total_time'][-1:]

            self.details_hist['G_T_Comp_im'] = self.details_hist['G_T_Comp_im'][-1:]
            self.details_hist['G_T_BCE_fake_real'] = self.details_hist['G_T_BCE_fake_real'][-1:]
            self.details_hist['G_T_Cycle'] = self.details_hist['G_T_Cycle'][-1:]
            self.details_hist['G_zCR'] = self.details_hist['G_zCR'][-1:]

            self.details_hist['G_V_Comp_im'] = self.details_hist['G_V_Comp_im'][-1:]
            self.details_hist['G_V_BCE_fake_real'] = self.details_hist['G_V_BCE_fake_real'][-1:]
            self.details_hist['G_V_Cycle'] = self.details_hist['G_V_Cycle'][-1:]

            self.details_hist['D_T_BCE_fake_real_R'] = self.details_hist['D_T_BCE_fake_real_R'][-1:]
            self.details_hist['D_T_BCE_fake_real_F'] = self.details_hist['D_T_BCE_fake_real_F'][-1:]
            self.details_hist['D_zCR'] = self.details_hist['D_zCR'][-1:]
            self.details_hist['D_bCR'] = self.details_hist['D_bCR'][-1:]

            self.details_hist['D_V_BCE_fake_real_R'] = self.details_hist['D_V_BCE_fake_real_R'][-1:]
            self.details_hist['D_V_BCE_fake_real_F'] = self.details_hist['D_V_BCE_fake_real_F'][-1:]
            ##Para poder tomar el promedio por epoch
            iterIniTrain = 1
            iterFinTrain = 1

            iterIniValidation = 1
            iterFinValidation = 1

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            if epoch % 10 == 0:
                self.save(str(epoch))
                with torch.no_grad():
                    if self.visdom:
                      self.visualize_results(epoch, dataprint=self.dataprint, visual=visImages)
                      self.visualize_results(epoch, dataprint=self.dataprint_test, visual=visImagesTest)
                    else:
                      imageName = self.model_name + '_' + 'Train' + '_' + str(self.seed) + '_' + str(epoch)
                      self.visualize_results(epoch, dataprint=self.dataprint, name= imageName)
                      self.visualize_results(epoch, dataprint=self.dataprint_test, name= imageName)


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        #utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                         self.epoch)
        #utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, dataprint, visual="", name= "test"):
        with torch.no_grad():
            self.G.eval()

            #if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            #    os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

            # print("sample z: ",self.sample_z_,"sample y:", self.sample_y_)

            ##Podria hacer un loop
            # .zfill(4)
            #newSample = None
            #print(dataprint.shape)

            #newSample = torch.tensor([])

            #se que es ineficiente pero lo hago cada 10 epoch nomas
            newSample = []
            iter = 1
            for x_im,x_dep in zip(dataprint.get('x_im'), dataprint.get('x_dep')):
                if (iter > self.cantImages):
                    break

                #x_im = (x_im + 1) / 2
                #imgX = transforms.ToPILImage()(x_im)
                #imgX.show()

                x_im_input = x_im.repeat(2, 1, 1, 1)
                x_dep_input = x_dep.repeat(2, 1, 1, 1)

                sizeImage = x_im.shape[2]

                sample_y_ = torch.zeros((self.class_num, 1, sizeImage, sizeImage))
                for i in range(self.class_num):
                    if(int(i % self.class_num) == 1):
                        sample_y_[i] = torch.ones(( 1, sizeImage, sizeImage))

                if self.gpu_mode:
                    sample_y_, x_im_input, x_dep_input = sample_y_.cuda(), x_im_input.cuda(), x_dep_input.cuda()

                G_im, G_dep = self.G(sample_y_, x_im_input, x_dep_input)

                newSample.append(x_im.squeeze(0))
                newSample.append(x_dep.squeeze(0).expand(3, -1, -1))



                if self.wiggle:
                    im_aux, im_dep_aux = G_im, G_dep
                    for i in range(0, 2):
                        index = i
                        for j in range(0, self.wiggleDepth):

                            # print(i,j)

                            if (j == 0 and i == 1):
                                # para tomar el original
                                im_aux, im_dep_aux = G_im, G_dep
                                newSample.append(G_im.cpu()[0].squeeze(0))
                                newSample.append(G_im.cpu()[1].squeeze(0))
                            elif (i == 1):
                                # por el problema de las iteraciones proximas
                                index = 0

                            # imagen generada


                            x = im_aux[index].unsqueeze(0)
                            x_dep = im_dep_aux[index].unsqueeze(0)

                            y = sample_y_[i].unsqueeze(0)

                            if self.gpu_mode:
                                y, x, x_dep = y.cuda(), x.cuda(), x_dep.cuda()

                            im_aux, im_dep_aux = self.G(y, x, x_dep)

                            newSample.append(im_aux.cpu()[0])
                else:

                    newSample.append(G_im.cpu()[0])
                    newSample.append(G_im.cpu()[1])
                    newSample.append(G_dep.cpu()[0].expand(3, -1, -1))
                    newSample.append(G_dep.cpu()[1].expand(3, -1, -1))
                    # sadadas

                iter+=1

            if self.visdom:
                visual.plot(epoch, newSample, int(len(newSample) /self.cantImages))
            else:
                utils.save_wiggle(newSample, self.cantImages, name)
        ##TENGO QUE HACER QUE SAMPLES TENGAN COMO MAXIMO self.class_num * self.class_num

        # utils.save_images(newSample[:, :, :, :], [image_frame_dim * cantidadIm , image_frame_dim * (self.class_num+2)],
        #                  self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%04d' % epoch + '.png')

    def show_plot_images(self, images, cols=1, titles=None):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        # assert ((titles is None) or (len(images) == len(titles)))
        n_images = len(images)
        if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            # print(image)
            image = (image + 1) * 255.0
            # print(image)
            # new_im = Image.fromarray(image)
            # print(new_im)
            if image.ndim == 2:
                plt.gray()
            # print("spi imshape ", image.shape)
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        plt.show()

    def joinImages(self, data):
        nData = []
        for i in range(self.class_num):
            nData.append(data)
        nData = np.array(nData)
        nData = torch.tensor(nData.tolist())
        nData = nData.type(torch.FloatTensor)

        return nData

    def save(self, epoch=''):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(),
                   os.path.join(save_dir, self.model_name + '_' + self.seed + '_' + epoch + '_G.pkl'))
        if not self.onlyGen:
          torch.save(self.D.state_dict(),
                   os.path.join(save_dir, self.model_name + '_' + self.seed + '_' + epoch + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history_ '+self.seed+'.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_' + self.seed_load + '_G.pkl')))
        if not self.wiggle:
            self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_' + self.seed_load + '_D.pkl')))

    def wiggleEf(self):
        seed, epoch = self.seed_load.split('_')
        if self.visdom:
            visWiggle = utils.VisdomImagePlotter(env_name='Cobo_depth_wiggle_' + seed)
            self.visualize_results(epoch=epoch, dataprint=self.dataprint_test, visual=visWiggle)
        else:
            self.visualize_results(epoch=epoch, dataprint=self.dataprint_test, visual=None, name = self.seed_load)

    def rearrengeData(self, Data):

        outpu = []
        for frame in Data:
            for camaraActual in range(self.nCameras):
                for camaraComparada in range(self.nCameras):

                    if camaraActual != camaraComparada:
                        # print("frame", frame)
                        valor_cambio = float((camaraComparada - camaraActual))

                        s = np.array([frame[2 * camaraActual].numpy(), frame[2 * camaraActual + 1].numpy(),
                                      frame[2 * camaraComparada].numpy(), valor_cambio,
                                      frame[2 * camaraComparada + 1].numpy()])
                        # print(s)
                        # outpu[camaraComparada + camaraActual*(self.nCameras+1)] = s
                        outpu.append(s)
        output = np.array(outpu)
        np.random.shuffle(output)  # para que no de 1 y uno
        return output[:, 0], output[:, 1], output[:, 2], output[:, 3], output[:, 4]
