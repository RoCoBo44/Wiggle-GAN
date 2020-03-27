import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.cuda as cu
import torch.optim as optim
from utils import augmentData
from PIL import Image
from dataloader import dataloader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from datetime import date
from statistics import mean
from architectures import depth_generator, depth_discriminator, depth_generator_UNet, discriminator_UNet, depth_discriminator_UNet


class MyACGAN(object):
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
        self.z_dim = 40
        self.class_num = (args.cameras - 1) * 2 #un calculo que hice en paint
        self.sample_num = self.class_num ** 2
        self.imageDim = args.imageDim
        self.epochVentaja = args.epochV
        self.cantImages = args.cIm
        random.seed(time.time())
        today = date.today()
        self.seed = str(random.randint(0,99999))
        self.seed_load = args.seedLoad
        self.toLoad = args.load

        self.vis = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visEpoch = utils.VisdomLineTwoPlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visImages = utils.VisdomImagePlotter(env_name='Cobo_depth_Images_' + str(today) + '_' + self.seed)

        self.visLossGTest = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visLossGValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)

        self.visLossDTest = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visLossDValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim, split='train', trans=True)

        self.data_Validation = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim, split='validation')

        self.dataprint = next(iter(self.data_Validation))[0:self.cantImages]  # Para agarrar varios
        #print(self.dataprint.shape)

        self.batch_size = self.batch_size * self.nCameras * (self.nCameras - 1) ## EXPLICADO EN VIDEO es por los frames
        data = self.data_loader.__iter__().__next__()[0]

        # networks init

        ## estoy muy perdido de como seria la entrada, para mi seria el anchox el alto de cada imagen pero tampoco se como la red se conecta Y alto y acho
        self.G = depth_generator_UNet(input_dim=6, output_dim=3, input_shape=data.shape, class_num=self.class_num,zdim = self.z_dim, height = data.shape[2], width = data.shape[3])
        #Ese 2 del input es porque es blanco y negro (imINICIO+imANGULO)
        self.D = depth_discriminator_UNet(input_dim=3, output_dim=1, input_shape=data.shape, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
            self.L1 = nn.L1Loss().cuda()
            self.MSE = nn.MSELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()
            self.MSE = nn.MSELoss()
            self.L1 = nn.L1Loss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

        if (self.toLoad):
            self.load()


    def train(self):
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
        self.details_hist['G_T_CE_Class'] = []

        self.details_hist['G_V_Comp_im'] = []
        self.details_hist['G_V_BCE_fake_real'] = []
        self.details_hist['G_V_CE_Class'] = []

        self.details_hist['D_T_CE_Class_R'] = []
        self.details_hist['D_T_BCE_fake_real_R'] = []
        self.details_hist['D_T_CE_Class_F'] = []
        self.details_hist['D_T_BCE_fake_real_F'] = []

        self.details_hist['D_V_CE_Class_R'] = []
        self.details_hist['D_V_BCE_fake_real_R'] = []
        self.details_hist['D_V_CE_Class_F'] = []
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

        #print("self.batch_size",self.batch_size)
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()
        #print("y_real_",self.y_real_)
        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):

            if (epoch < self.epochVentaja):
                ventaja = True
            else:
                ventaja = False

            self.G.train()
            epoch_start_time = time.time()


            for iter,data in enumerate(self.data_loader): #Cambiar con el dataset, agarra por batch size

                #Aumento mi data
                x_im,x_dep,y_im,y_ = self.rearrengeData(data)
                x_im_aug, y_im_aug = augmentData(x_im, y_im)

                #para que usen Tensor
                x_im  = torch.tensor(x_im.tolist())
                y_im  = torch.tensor(y_im.tolist())
                x_dep = torch.tensor(x_dep.tolist())
                y_    = torch.tensor(y_.tolist())
                x_im_aug = torch.tensor(x_im_aug.tolist())
                y_im_aug = torch.tensor(y_im_aug.tolist())

                x_im  = x_im.type(torch.FloatTensor)
                y_im  = y_im.type(torch.FloatTensor)
                x_dep = x_dep.type(torch.FloatTensor)
                y_    = y_.type(torch.FloatTensor)
                x_im_aug = x_im_aug.type(torch.FloatTensor)
                y_im_aug = y_im_aug.type(torch.FloatTensor)

                # x_im  = imagenes normales
                # x_dep = profundidad de images
                # y_im  = imagen con el angulo cambiado
                # y_    = angulo de la imagen = tengo que tratar negativos


                y_ = y_ + (self.nCameras -1)
                y_ = torch.Tensor(list(map(lambda x: int(x-1) if (x > 0)else int(x), y_)))
                #x_im  = torch.Tensor(list(x_im))
                #x_dep = torch.Tensor(x_dep)
                #y_im  = torch.Tensor(y_im)
                #print(y_.shape[0])
                if iter == self.data_loader.dataset.__len__() * self.nCameras * (self.nCameras - 1)// self.batch_size:
                    #print ("Break")
                    break
                z_ = torch.rand((y_.shape[0], self.z_dim))
                #print (y_.type(torch.LongTensor).unsqueeze(1))
                y_vec_ = torch.zeros((y_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)

                #print("y_vec_",y_vec_)
                #print ("z_",z_)

                if self.gpu_mode:
                    x_im, z_, y_vec_, y_im, x_dep, x_im_aug, y_im_aug = x_im.cuda(), z_.cuda(), y_vec_.cuda(),y_im.cuda(),x_dep.cuda(), x_im_aug.cuda(), y_im_aug.cuda()
                # update D network

                if not ventaja:

                    self.D_optimizer.zero_grad()
                    """                    
                    D_real, D_clase_real= self.D(y_im, x_im)  ## Es la funcion forward `` g(z) x

                    D_real_loss = self.BCE_loss(D_real, self.y_real_)
                    D_clase_real_loss = self.CE_loss(D_clase_real, torch.max(y_vec_, 1)[1])

                    G_ = self.G(z_, y_vec_, x_im, x_dep)
                    D_fake, D_clase_fake = self.D(G_, x_im)

                    D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                    D_clase_fake_loss = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1])

                    D_loss = (D_real_loss + D_fake_loss + D_clase_real_loss + D_clase_fake_loss)

                    self.train_hist['D_loss_train'].append(D_loss.item())
                    self.details_hist['D_T_CE_Class_R'].append(D_clase_real_loss.item())
                    self.details_hist['D_T_BCE_fake_real_R'].append(D_real_loss.item())
                    self.details_hist['D_T_CE_Class_F'].append(D_clase_fake_loss.item())
                    self.details_hist['D_T_BCE_fake_real_F'].append(D_fake_loss.item())
                    self.visLossDTest.plot('Discriminator_losses',
                                           ['D_T_CE_Class_R', 'D_T_BCE_fake_real_R', 'D_T_CE_Class_F',
                                            'D_T_BCE_fake_real_F'], 'train', self.details_hist)
                    """
                    # Real Images
                    D_real, D_clase_real= self.D(y_im, x_im)  ## Es la funcion forward `` g(z) x

                    # Fake Images
                    G_ = self.G(z_, y_vec_, x_im, x_dep)
                    D_fake, D_clase_fake= self.D(G_, x_im)

                    # Fake Augmented Images bCR

                    x_im_aug_bCR, G_aug_bCR = augmentData(x_im.data.cpu().numpy(), G_.data.cpu().numpy())
                    x_im_aug_bCR = torch.tensor(x_im_aug_bCR.tolist())
                    G_aug_bCR = torch.tensor(G_aug_bCR.tolist())
                    x_im_aug_bCR = x_im_aug_bCR.type(torch.FloatTensor)
                    G_aug_bCR = G_aug_bCR.type(torch.FloatTensor)
                    if self.gpu_mode:
                        G_aug_bCR, x_im_aug_bCR= G_aug_bCR.cuda(), x_im_aug_bCR.cuda()

                    D_fake_bCR, D_fake_bCR = self.D(G_aug_bCR, x_im_aug_bCR)
                    D_real_bCR, D_real_bCR = self.D(y_im_aug, x_im_aug)

                    # Fake Augmented Images zCR
                    G_aug_zCR = self.G(z_, y_vec_, x_im_aug, x_dep)
                    D_fake_aug_zCR , D_clase_fake_aug = self.D(G_aug_zCR, x_im_aug)

                    # Losses
                    #  GAN Loss
                    D_loss_real_fake = torch.mean(D_fake) - torch.mean(D_real)
                    #  Class Loss
                    D_loss_Class = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1]) + self.CE_loss(D_clase_real, torch.max(y_vec_, 1)[1])
                    #  bCR Loss
                    D_loss_real = self.MSE(D_real, D_real_bCR )
                    D_loss_fake = self.MSE(D_fake, D_fake_bCR )
                    D_bCR = (D_loss_real + D_loss_fake)*0.5 # ACA EN EL PAPER USA UN FACTOR PERO NO SE COMO AGRGARLO, igual que zCR

                    #  zCR Loss
                    D_zCR = self.MSE(D_fake, D_fake_aug_zCR )*0.5
                    D_loss = D_zCR + D_loss_Class + D_loss_real_fake + D_bCR

                    self.train_hist['D_loss_train'].append(D_loss.item())
                    self.details_hist['D_T_CE_Class_R'].append(D_loss_Class.item())
                    self.details_hist['D_T_BCE_fake_real_R'].append(D_loss_real_fake.item())
                    self.visLossDTest.plot('Discriminator_losses',
                                           ['D_T_CE_Class_R', 'D_T_BCE_fake_real_R'], 'train', self.details_hist)

                    D_loss.backward()
                    self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_, x_im, x_dep)

                if not ventaja:

                    """
                    D_fake, C_fake = self.D(G_, x_im)

                    G_loss_BCE = self.BCE_loss(D_fake, self.y_real_)
                    C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])
                    G_loss_MSE = self.MSE(G_, y_im)
                    G_loss = C_fake_loss + G_loss_BCE + G_loss_MSE


                    self.details_hist['G_T_MSE_im'].append(G_loss_MSE.item())
                    self.details_hist['G_T_BCE_fake_real'].append(G_loss_BCE.item())
                    self.details_hist['G_T_CE_Class'].append(C_fake_loss.item())
                    """
                    # Fake images
                    D_fake, D_clase_fake = self.D(G_, x_im)

                    G_aug = self.G(z_, y_vec_, x_im_aug, x_dep)

                    G_loss = -torch.mean(D_fake)
                    self.details_hist['G_T_BCE_fake_real'].append(G_loss.item())
                    G_loss_Comp = self.MSE(G_, y_im)
                    G_loss_Comp_Aug = self.MSE(G_aug, y_im_aug)
                    G_loss_Dif_Comp = G_loss_Comp + G_loss_Comp_Aug
                    G_zCR = -self.MSE(G_, G_aug)
                    C_fake_loss = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1])
                    G_loss += G_zCR + G_loss_Dif_Comp + C_fake_loss

                    self.details_hist['G_T_Comp_im'].append(G_loss_Dif_Comp.item())
                    self.details_hist['G_T_CE_Class'].append(C_fake_loss.item())

                else:
                    G_loss = self.L1(G_, y_im)
                    self.details_hist['G_T_Comp_im'].append(G_loss.item())
                    self.details_hist['G_T_BCE_fake_real'].append(0)
                    self.details_hist['G_T_CE_Class'].append(0)

                self.train_hist['G_loss_train'].append(G_loss.item())
                iterFinTrain += 1

                G_loss.backward()
                self.G_optimizer.step()

                self.visLossGTest.plot('Generator_losses', ['G_T_Comp_im', 'G_T_BCE_fake_real', 'G_T_CE_Class'],
                                       'train', self.details_hist)
                #self.visLossGTest.plot('Generator_losses', ['G_T_MSE_im','G_T_BCE_fake_real','G_T_CE_Class'], 'train', self.details_hist)
                self.vis.plot('loss',['D_loss_train','G_loss_train'], 'train', self.train_hist)

            ##################Validation#####################################
            for iter, data in enumerate(self.data_Validation):

                # Aumento mi data
                x_im, x_dep, y_im, y_ = self.rearrengeData(data)

                # para que usen Tensor
                x_im = torch.tensor(x_im.tolist())
                y_im = torch.tensor(y_im.tolist())
                x_dep = torch.tensor(x_dep.tolist())
                y_ = torch.tensor(y_.tolist())

                x_im = x_im.type(torch.FloatTensor)
                y_im = y_im.type(torch.FloatTensor)
                x_dep = x_dep.type(torch.FloatTensor)
                y_ = y_.type(torch.FloatTensor)
                # x_im  = imagenes normales
                # x_dep = profundidad de images
                # y_im  = imagen con el angulo cambiado
                # y_    = angulo de la imagen = tengo que tratar negativos

                y_ = y_ + (self.nCameras - 1)
                y_ = torch.Tensor(list(map(lambda x: int(x - 1) if (x > 0) else int(x), y_)))
                # x_im  = torch.Tensor(list(x_im))
                # x_dep = torch.Tensor(x_dep)
                # y_im  = torch.Tensor(y_im)
                #print(y_.shape[0])
                if iter == self.data_Validation.dataset.__len__() * self.nCameras * (self.nCameras - 1) // self.batch_size:
                    #print ("Break")
                    break
                z_ = torch.rand((y_.shape[0], self.z_dim))
                #print (y_.type(torch.LongTensor).unsqueeze(1))
                y_vec_ = torch.zeros((y_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1).long()

                #print("y_vec_", y_vec_)
                #print ("z_", z_)

                if self.gpu_mode:
                    x_im, z_, y_vec_, y_im, x_dep = x_im.cuda(), z_.cuda(), y_vec_.cuda(), y_im.cuda(), x_dep.cuda()
                # update D network

                if not ventaja:
                    """
                    D_real, D_clase_real = self.D(y_im, x_im)  ## Es la funcion forward `` g(z) x

                    D_real_loss = self.BCE_loss(D_real, self.y_real_)
                    D_clase_real_loss = self.CE_loss(D_clase_real, torch.max(y_vec_, 1)[1])

                    G_ = self.G(z_, y_vec_, x_im, x_dep)

                    D_fake, D_clase_fake = self.D(G_, x_im)

                    D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                    D_clase_fake_loss = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1])

                    D_loss = (D_real_loss + D_fake_loss + D_clase_real_loss + D_clase_fake_loss)

                    self.train_hist['D_loss_Validation'].append(D_loss.item())
                    self.details_hist['D_V_CE_Class_R'].append(D_clase_real_loss.item())
                    self.details_hist['D_V_BCE_fake_real_R'].append(D_real_loss.item())
                    self.details_hist['D_V_CE_Class_F'].append(D_clase_fake_loss.item())
                    self.details_hist['D_V_BCE_fake_real_F'].append(D_fake_loss.item())
                    self.visLossDValidation.plot('Discriminator_losses',
                                                 ['D_V_CE_Class_R', 'D_V_BCE_fake_real_R', 'D_V_CE_Class_F',
                                                  'D_V_BCE_fake_real_F'], 'Validation', self.details_hist)
                    """
                    # Real Images
                    D_real, D_clase_real= self.D(y_im, x_im)  ## Es la funcion forward `` g(z) x

                    # Fake Images
                    G_ = self.G(z_, y_vec_, x_im, x_dep)
                    D_fake, D_clase_fake= self.D(G_, x_im)

                    # Losses
                    #  GAN Loss
                    D_loss_real_fake = torch.mean(D_fake) - torch.mean(D_real)
                    #  Class Loss
                    D_loss_Class = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1]) + self.CE_loss(D_clase_real, torch.max(y_vec_, 1)[1])

                    D_loss = D_loss_Class + D_loss_real_fake

                    self.train_hist['D_loss_Validation'].append(D_loss.item())
                    self.details_hist['D_V_CE_Class_R'].append(D_loss_Class.item())
                    self.details_hist['D_V_BCE_fake_real_R'].append(D_loss_real_fake.item())
                    self.visLossDValidation.plot('Discriminator_losses',
                                           ['D_V_CE_Class_R', 'D_V_BCE_fake_real_R'], 'Validation', self.details_hist)

                # update G network

                G_ = self.G(z_, y_vec_, x_im, x_dep)

                if not ventaja:
                    """
                    D_fake, C_fake = self.D(G_, x_im)

                    G_loss_BCE = self.BCE_loss(D_fake, self.y_real_)
                    C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])
                    G_loss_MSE = self.MSE(G_, y_im)
                    G_loss = C_fake_loss + G_loss_BCE + G_loss_MSE

                    self.details_hist['G_V_MSE_im'].append(G_loss_MSE.item())
                    self.details_hist['G_V_BCE_fake_real'].append(G_loss_BCE.item())
                    self.details_hist['G_V_CE_Class'].append(C_fake_loss.item())
                    """

                    #Fake images
                    D_fake, D_clase_fake = self.D(G_, x_im)

                    # Tener en cuenta que el loss de la imagen como no hay augmentation, se calcula de otra forma
                    G_loss = -torch.mean(D_fake)
                    self.details_hist['G_V_BCE_fake_real'].append(G_loss.item())
                    G_loss_Comp = self.MSE(G_, y_im)
                    C_fake_loss = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1])
                    G_loss += G_loss_Comp + C_fake_loss

                    self.details_hist['G_V_Comp_im'].append(G_loss_Comp.item())
                    self.details_hist['G_V_CE_Class'].append(C_fake_loss.item())

                else:
                    G_loss = self.L1(G_, y_im)
                    self.details_hist['G_V_Comp_im'].append(G_loss.item())
                    self.details_hist['G_V_BCE_fake_real'].append(0)
                    self.details_hist['G_V_CE_Class'].append(0)

                self.train_hist['G_loss_Validation'].append(G_loss.item())
                iterFinValidation += 1
                self.visLossGValidation.plot('Generator_losses', ['G_V_Comp_im', 'G_V_BCE_fake_real', 'G_V_CE_Class'],
                                       'Validation', self.details_hist)
                self.visValidation.plot('loss',['D_loss_Validation','G_loss_Validation'], 'Validation', self.train_hist)

            ##Vis por epoch

            if ventaja:
                self.epoch_hist['D_loss_train'].append(0)
                self.epoch_hist['D_loss_Validation'].append(0)
            else:
                inicioTr = (epoch  - self.epochVentaja) * (iterFinTrain - iterIniTrain)
                inicioTe = (epoch  - self.epochVentaja) * (iterFinValidation - iterIniValidation)
                self.epoch_hist['D_loss_train'].append(mean(self.train_hist['D_loss_train'][ inicioTr: -1 ]))
                self.epoch_hist['D_loss_Validation'].append(mean(self.train_hist['D_loss_Validation'][ inicioTe: -1 ]))


            self.epoch_hist['G_loss_train'].append(mean(self.train_hist['G_loss_train'][iterIniTrain:iterFinTrain]))
            self.epoch_hist['G_loss_Validation'].append(mean(self.train_hist['G_loss_Validation'][iterIniValidation:iterFinValidation]))

            self.visEpoch.plot('epoch',epoch, ['D_loss_train','G_loss_train','D_loss_Validation','G_loss_Validation'] , self.epoch_hist)

            ##cambio de iters
            iterIniValidation = iterFinValidation
            iterIniTrain = iterFinTrain

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

            if epoch % 50 == 0:
                self.save(str(epoch))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)


    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)


        #print("sample z: ",self.sample_z_,"sample y:", self.sample_y_)

        ##Podria hacer un loop
        #.zfill(4)
        cantidadIm = self.cantImages
        newSample = None
        image_frame_dim = int(np.floor(np.sqrt(self.class_num)))
        for set in self.dataprint:
            data1 = set[0]
            #print (data1.shape)
            data = data1
            realIm = data1
            nData = self.joinImages(data.numpy())
            #print (data.shape)

            data2 = set[1]
            #print (data2.shape)
            dep = data2
            nData2 = self.joinImages(data2.numpy())
            #print (data.shape)



            #print("nData shape: ", nData.shape)

            if self.gpu_mode:
                data = nData.cuda()
                data2 = nData2.cuda()

            self.sample_z_ = torch.rand(cantidadIm * self.class_num, self.z_dim)
            #print ("self.sample_z_.shape", self.sample_z_.shape)
            # for j in range(1, self.class_num):
            #    self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

            temp = torch.zeros((self.class_num, 1))
            for i in range(self.class_num):
                temp[i, 0] = i
            #print(temp)

            temp_y = torch.zeros((self.class_num * cantidadIm, 1))
            #print ("tempy", temp_y)
            for i in range(len(temp)):
                temp_y[i] = int(i % self.class_num)
            #print ("tempy", temp_y)
            self.sample_y_ = torch.zeros((self.class_num * cantidadIm, self.class_num)).scatter_(1, temp_y.type(
                torch.LongTensor), 1)

            #print ("self.sample_y_", self.sample_y_)
            if self.gpu_mode:
                self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

            if fix:
                """ fixed noise """
                samples = self.G(self.sample_z_, self.sample_y_, data, data2)
            else:
                """ random noise """
                sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (
                self.batch_size, 1)).type(torch.LongTensor), 1)
                sample_z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

                samples = self.G(sample_z_, sample_y_, data, data2)

            if self.gpu_mode:
                samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1) #los valores son la posicion, corre lo segudno a lo ultimo
            else:
                samples = samples.data.numpy().transpose(0, 2, 3, 1)


            ## TRATO DE PASAR IMAGEN ORIGINAL
            #print("realIm",realIm.shape)
            realIm = torch.tensor(realIm.tolist())
            realIm = realIm.type(torch.FloatTensor)
            realIm = realIm.unsqueeze(0)
            realIm = realIm.cpu().data.numpy().transpose(0, 2, 3, 1)
            realIm = np.array(realIm)


            dep = torch.tensor(dep.tolist())
            dep = dep.type(torch.FloatTensor)
            dep = dep.unsqueeze(0)
            dep = dep.cpu().data.numpy().transpose(0, 2, 3, 1)
            dep = np.array(dep)

            samples = np.concatenate((samples, realIm))
            samples = np.concatenate((samples, dep))
            if newSample is None:
                newSample = samples
            else:
                newSample = np.concatenate((newSample,samples))

        newSample = (newSample + 1) / 2

        self.visImages.plot(epoch,newSample,4)
        ##TENGO QUE HACER QUE SAMPLES TENGAN COMO MAXIMO self.class_num * self.class_num

        utils.save_images(newSample[:, :, :, :], [image_frame_dim * cantidadIm , image_frame_dim * (self.class_num+2)],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%04d' % epoch + '.png')




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
            #print(image)
            image = (image + 1) * 255.0
            #print(image)
            # new_im = Image.fromarray(image)
            # print(new_im)
            if image.ndim == 2:
                plt.gray()
            #print("spi imshape ", image.shape)
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

    def save(self, epoch = ''):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_' +self.seed + '_' + epoch + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_' +self.seed + '_' + epoch + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_' +self.seed_load + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_' +self.seed_load + '_D.pkl')))

    def rearrengeData(self, Data):


            outpu = []
            for frame in Data:
                for camaraActual in range(self.nCameras):
                    for camaraComparada in range(self.nCameras):

                        if camaraActual != camaraComparada:
                            #print("frame", frame)
                            valor_cambio = camaraComparada - camaraActual

                            s = np.array([frame[2 * camaraActual].numpy(), frame[2 * camaraActual + 1].numpy(), frame[2 * camaraComparada].numpy(), valor_cambio])
                            #print(s)
                            # outpu[camaraComparada + camaraActual*(self.nCameras+1)] = s
                            outpu.append(s)

            output = np.array(outpu)
            np.random.shuffle(output) #para que no de 1 y uno
           # print (output)
            return output[:,0], output[:,1],output[:,2],output[:,3]


