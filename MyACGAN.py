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
from architectures import depth_generator, depth_discriminator, depth_generator_UNet, discriminator_UNet, \
    depth_discriminator_UNet


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
        self.class_num = (args.cameras - 1) * 2  # un calculo que hice en paint
        self.sample_num = self.class_num ** 2
        self.imageDim = args.imageDim
        self.epochVentaja = args.epochV
        self.cantImages = args.cIm
        self.visdom = args.visdom

        random.seed(time.time())
        today = date.today()
        self.seed = str(random.randint(0, 99999))
        self.seed_load = args.seedLoad
        self.toLoad = args.load

        self.zGenFactor = args.zGF
        self.zDisFactor = args.zDF
        self.bFactor = args.bF

        self.expandGen = args.expandGen
        self.expandDis = args.expandDis

        self.wiggle = args.wiggle
        self.wiggleDepth = args.wiggleDepth

        self.vis = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visValidation = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visEpoch = utils.VisdomLineTwoPlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visImages = utils.VisdomImagePlotter(env_name='Cobo_depth_Images_' + str(today) + '_' + self.seed)
        self.visImagesTest = utils.VisdomImagePlotter(env_name='Cobo_depth_ImagesTest_' + str(today) + '_' + self.seed)

        self.visLossGTest = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visLossGValidation = utils.VisdomLinePlotter(
            env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)

        self.visLossDTest = utils.VisdomLinePlotter(env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)
        self.visLossDValidation = utils.VisdomLinePlotter(
            env_name='Cobo_depth_Train-Plots_' + str(today) + '_' + self.seed)

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim, split='train',
                                      trans=True)

        self.data_Validation = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim,
                                          split='validation')
        self.data_Test = dataloader(self.dataset, self.input_size, self.batch_size, self.imageDim, split='test')

        self.dataprint = self.data_Validation.__iter__().__next__()  # next(iter(self.data_Validation))[0:self.cantImages]  # Para agarrar varios
        self.dataprint_test = self.data_Test.__iter__().__next__()

        self.batch_size = self.batch_size #* self.nCameras * (self.nCameras - 1)  ## EXPLICADO EN VIDEO es por los frames
        data = self.data_loader.__iter__().__next__().get('x_im')

        # networks init

        self.G = depth_generator_UNet(input_dim=4, output_dim=3, class_num=self.class_num, expand_net=self.expandGen)
        # Ese 2 del input es porque es blanco y negro (imINICIO+imANGULO)
        self.D = depth_discriminator_UNet(input_dim=3, output_dim=1, input_shape=data.shape, class_num=self.class_num,
                                          expand_net=self.expandDis)
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
        self.details_hist['G_zCR'] = []

        self.details_hist['G_V_Comp_im'] = []
        self.details_hist['G_V_BCE_fake_real'] = []
        self.details_hist['G_V_CE_Class'] = []

        self.details_hist['D_T_CE_Class_R'] = []
        self.details_hist['D_T_BCE_fake_real_R'] = []
        self.details_hist['D_T_CE_Class_F'] = []
        self.details_hist['D_T_BCE_fake_real_F'] = []
        self.details_hist['D_zCR'] = []
        self.details_hist['D_bCR'] = []

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

        self.D.train()
        print('training start!!')
        start_time = time.time()

        maxIter = self.data_loader.dataset.__len__() // self.batch_size
        maxIterVal = self.data_Validation.dataset.__len__() // self.batch_size

        for epoch in range(self.epoch):

            if (epoch < self.epochVentaja):
                ventaja = True
            else:
                ventaja = False

            self.G.train()
            epoch_start_time = time.time()

            for iter, data in enumerate(self.data_loader):  # Cambiar con el dataset, agarra por batch size

                x_im = data.get('x_im')
                x_dep = data.get('x_dep')
                y_im = data.get('y_im')
                y_dep = data.get('y_dep')
                y_ = data.get('y_')
                #{'x_im': x1, 'x_dep': x1_dep, 'y_im': x2, 'y_dep': x2_dep, 'y_': torch.tensor(1)}



                # Aumento mi data
                x_im_aug, y_im_aug = augmentData(x_im, y_im)
                x_im_vanilla = x_im

                # x_im  = imagenes normales
                # x_dep = profundidad de images
                # y_im  = imagen con el angulo cambiado
                # y_    = angulo de la imagen = tengo que tratar negativos

                #y_ = torch.Tensor(list(map(lambda x: int(x - 1) if (x > 0) else int(x), y_)))

                # x_im  = torch.Tensor(list(x_im))
                # x_dep = torch.Tensor(x_dep)
                # y_im  = torch.Tensor(y_im)
                # print(y_.shape[0])
                if iter >= maxIter:
                    # print ("Break")
                    break
                # print (y_.type(torch.LongTensor).unsqueeze(1))
                y_vec_ = torch.zeros((y_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1),1)

                # print("y_vec_",y_vec_)
                if self.gpu_mode:
                    x_im, y_vec_, y_im, x_dep, x_im_aug, y_im_aug, y_dep = x_im.cuda(), y_vec_.cuda(), y_im.cuda(), x_dep.cuda(), x_im_aug.cuda(), y_im_aug.cuda(), y_dep.cuda()
                # update D network

                if not ventaja:

                    self.D_optimizer.zero_grad()

                    # Real Images
                    D_real, D_clase_real = self.D(y_im, x_im, y_dep)  ## Es la funcion forward `` g(z) x

                    # Fake Images
                    G_, G_dep = self.G( y_vec_, x_im, y_dep)
                    D_fake, D_clase_fake = self.D(G_, x_im, G_dep)

                    # Fake Augmented Images bCR
                    x_im_aug_bCR, G_aug_bCR = augmentData(x_im_vanilla, G_.data.cpu())
                    #x_im_aug_bCR = torch.from_numpy(x_im_aug_bCR).type(torch.FloatTensor)
                    #G_aug_bCR = torch.from_numpy(G_aug_bCR).type(torch.FloatTensor)

                    #x_im_aug_bCR = x_im_aug_bCR.type(torch.FloatTensor)
                    #G_aug_bCR = G_aug_bCR.type(torch.FloatTensor)
                    if self.gpu_mode:
                        G_aug_bCR, x_im_aug_bCR = G_aug_bCR.cuda(), x_im_aug_bCR.cuda()

                    D_fake_bCR, D_fake_bCR = self.D(G_aug_bCR, x_im_aug_bCR, G_dep)
                    D_real_bCR, D_real_bCR = self.D(y_im_aug, x_im_aug, y_dep)

                    # Fake Augmented Images zCR
                    G_aug_zCR, G_dep_aug_zCR = self.G(y_vec_, x_im_aug, x_dep)
                    D_fake_aug_zCR, D_clase_fake_aug = self.D(G_aug_zCR, x_im_aug, G_dep_aug_zCR)

                    # Losses
                    #  GAN Loss
                    D_loss_real_fake = torch.mean(D_fake) - torch.mean(D_real)
                    #  Class Loss
                    D_loss_Class = self.CE_loss(D_clase_real, torch.max(y_vec_, 1)[
                        1])  # self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1]) +
                    #  bCR Loss
                    D_loss_real = self.MSE(D_real, D_real_bCR)
                    D_loss_fake = self.MSE(D_fake, D_fake_bCR)
                    D_bCR = (D_loss_real + D_loss_fake) * self.bFactor  # ACA EN EL PAPER USA UN FACTOR PERO NO SE COMO AGRGARLO, igual que zCR

                    #  zCR Loss
                    D_zCR = self.MSE(D_fake, D_fake_aug_zCR) * self.zDisFactor
                    D_loss = D_zCR + D_loss_Class + D_loss_real_fake + D_bCR

                    self.train_hist['D_loss_train'].append(D_loss.item())
                    self.details_hist['D_T_CE_Class_R'].append(D_loss_Class.item())
                    self.details_hist['D_T_BCE_fake_real_R'].append(D_loss_real_fake.item())
                    self.details_hist['D_bCR'].append(D_bCR.item())
                    self.details_hist['D_zCR'].append(D_zCR.item())
                    self.visLossDTest.plot('Discriminator_losses',
                                           ['D_T_CE_Class_R', 'D_T_BCE_fake_real_R', 'D_bCR', 'D_zCR'], 'train',
                                           self.details_hist)

                    D_loss.backward()
                    self.D_optimizer.step()


                # update G network
                self.G_optimizer.zero_grad()

                G_, G_dep = self.G(y_vec_, x_im, x_dep)

                if not ventaja:

                    # Fake images augmented

                    G_aug, G_dep_aug = self.G(y_vec_, x_im_aug, x_dep)

                    D_fake_aug, D_clase_fake_aug = self.D(G_aug, x_im, G_dep_aug)

                    # Fake images
                    D_fake, D_clase_fake = self.D(G_, x_im, G_dep)

                    G_loss = -(torch.mean(D_fake) + torch.mean(D_fake_aug))/2
                    self.details_hist['G_T_BCE_fake_real'].append(G_loss.item())

                    # Class
                    C_fake_loss = (self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1]) +
                                   self.CE_loss(D_clase_fake_aug, torch.max(y_vec_, 1)[1])
                                   )/2

                    # loss between images
                    G_join = torch.cat((G_, G_dep), 1)
                    y_join = torch.cat((y_im, y_dep), 1)
                    y_aug_join = torch.cat((y_im_aug, y_dep), 1)
                    G_aug_join = torch.cat((G_aug, G_dep_aug), 1)


                    G_loss_Comp = self.L1(G_join, y_join)
                    G_loss_Comp_Aug = self.L1(G_aug_join, y_aug_join)
                    G_loss_Dif_Comp = G_loss_Comp + G_loss_Comp_Aug


                    # zCR
                    #G_zCR = -self.MSE(G_, G_aug) * self.zGenFactor



                    # dep
                    # G_loss_Comp_dep = self.L1(G_dep, y_im)
                    # G_loss_Comp_Aug_dep = self.L1(G_aug, y_im_aug)
                    # G_loss_Dif_Comp_dep = G_loss_Comp + G_loss_Comp_Aug

                    G_loss += G_loss_Dif_Comp + C_fake_loss


                    self.details_hist['G_T_Comp_im'].append(G_loss_Dif_Comp.item())
                    self.details_hist['G_T_CE_Class'].append(C_fake_loss.item())
                    self.details_hist['G_zCR'].append(0)


                else:
                    G_join = torch.cat((G_, G_dep), 1)
                    y_join = torch.cat((y_im, y_dep), 1)
                    G_loss = self.L1(G_join, y_join)
                    self.details_hist['G_T_Comp_im'].append(G_loss.item())
                    self.details_hist['G_T_BCE_fake_real'].append(0)
                    self.details_hist['G_T_CE_Class'].append(0)
                    self.details_hist['G_zCR'].append(0)

                G_loss.backward()
                self.G_optimizer.step()
                self.train_hist['G_loss_train'].append(G_loss.item())

                iterFinTrain += 1

                self.visLossGTest.plot('Generator_losses',
                                      ['G_T_Comp_im', 'G_T_BCE_fake_real', 'G_T_CE_Class', 'G_zCR'],
                                       'train', self.details_hist)

                self.vis.plot('loss', ['D_loss_train', 'G_loss_train'], 'train', self.train_hist)

            ##################Validation#####################################
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
                y_vec_ = torch.zeros((y_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1),
                                                                             1).long()


                # print("y_vec_", y_vec_)
                # print ("z_", z_)

                if self.gpu_mode:
                    x_im, y_vec_, y_im, x_dep, y_dep = x_im.cuda(), y_vec_.cuda(), y_im.cuda(), x_dep.cuda(), y_dep.cuda()
                # D network

                if not ventaja:
                    # Real Images
                    D_real, D_clase_real = self.D(y_im, x_im, y_dep)  ## Es la funcion forward `` g(z) x

                    # Fake Images
                    G_, G_dep = self.G(y_vec_, x_im, x_dep)
                    D_fake, D_clase_fake = self.D(G_, x_im, G_dep)
                    # Losses
                    #  GAN Loss
                    D_loss_real_fake = torch.mean(D_fake) - torch.mean(D_real)
                    #  Class Loss
                    D_loss_Class = self.CE_loss(D_clase_real, torch.max(y_vec_, 1)[
                        1])  # self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1]) +

                    D_loss = D_loss_Class + D_loss_real_fake

                    self.train_hist['D_loss_Validation'].append(D_loss.item())
                    self.details_hist['D_V_CE_Class_R'].append(D_loss_Class.item())
                    self.details_hist['D_V_BCE_fake_real_R'].append(D_loss_real_fake.item())
                    self.visLossDValidation.plot('Discriminator_losses',
                                                 ['D_V_CE_Class_R', 'D_V_BCE_fake_real_R'], 'Validation',
                                                 self.details_hist)

                # G network

                G_, G_dep = self.G(y_vec_, x_im, x_dep)

                if not ventaja:
                    # Fake images
                    D_fake, D_clase_fake = self.D(G_, x_im, G_dep)

                    # Tener en cuenta que el loss de la imagen como no hay augmentation, se calcula de otra forma
                    G_loss = -torch.mean(D_fake)
                    self.details_hist['G_V_BCE_fake_real'].append(G_loss.item())

                    G_join = torch.cat((G_, G_dep), 1)
                    y_join = torch.cat((y_im, y_dep), 1)
                    G_loss_Comp = self.L1(G_join, y_join)


                    C_fake_loss = self.CE_loss(D_clase_fake, torch.max(y_vec_, 1)[1])
                    G_loss += G_loss_Comp + C_fake_loss


                    self.details_hist['G_V_Comp_im'].append(G_loss_Comp.item())
                    self.details_hist['G_V_CE_Class'].append(C_fake_loss.item())

                else:
                    G_join = torch.cat((G_, G_dep), 1)
                    y_join = torch.cat((y_im, y_dep), 1)
                    G_loss = self.L1(G_join, y_join)
                    self.details_hist['G_V_Comp_im'].append(G_loss.item())
                    self.details_hist['G_V_BCE_fake_real'].append(0)
                    self.details_hist['G_V_CE_Class'].append(0)
                    
                self.train_hist['G_loss_Validation'].append(G_loss.item())


                iterFinValidation += 1
                self.visLossGValidation.plot('Generator_losses', ['G_V_Comp_im', 'G_V_BCE_fake_real', 'G_V_CE_Class'],
                                             'Validation', self.details_hist)
                self.visValidation.plot('loss', ['D_loss_Validation', 'G_loss_Validation'], 'Validation',
                                       self.train_hist)

            ##Vis por epoch

            if ventaja:
                self.epoch_hist['D_loss_train'].append(0)
                self.epoch_hist['D_loss_Validation'].append(0)
            else:
                inicioTr = (epoch - self.epochVentaja) * (iterFinTrain - iterIniTrain)
                inicioTe = (epoch - self.epochVentaja) * (iterFinValidation - iterIniValidation)
                self.epoch_hist['D_loss_train'].append(mean(self.train_hist['D_loss_train'][inicioTr: -1]))
                self.epoch_hist['D_loss_Validation'].append(mean(self.train_hist['D_loss_Validation'][inicioTe: -1]))

            self.epoch_hist['G_loss_train'].append(mean(self.train_hist['G_loss_train'][iterIniTrain:iterFinTrain]))
            self.epoch_hist['G_loss_Validation'].append(
                mean(self.train_hist['G_loss_Validation'][iterIniValidation:iterFinValidation]))

            self.visEpoch.plot('epoch', epoch,
                               ['D_loss_train', 'G_loss_train', 'D_loss_Validation', 'G_loss_Validation'],
                               self.epoch_hist)


            ## In order to load data
            #f = open('epochData.pkl', 'rb')
            #savedData = pickle.load(f)
            #f.close()

            ##cambio de iters
            iterIniValidation = iterFinValidation
            iterIniTrain = iterFinTrain

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            if epoch % 10 == 0:
                self.save(str(epoch))
                with torch.no_grad():
                    self.visualize_results(epoch, dataprint=self.dataprint, visual=self.visImages)
                    self.visualize_results(epoch, dataprint=self.dataprint_test, visual=self.visImagesTest)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        #utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
        #                         self.epoch)
        #utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, dataprint, visual, fix=True):
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


            temp_y = torch.zeros((self.class_num , 1))
            for i in range(self.class_num):
                temp_y[i] = int(i % self.class_num)
            sample_y_ = torch.zeros((self.class_num , self.class_num)).scatter_(1, temp_y.type(
                torch.LongTensor), 1)


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



        """""
        for set in dataprint:
            data1 = set[0]
            # print (data1.shape)
            data = data1
            realIm = data1
            nData = self.joinImages(data.numpy())
            # print (data.shape)

            data2 = set[1]
            # print (data2.shape)
            dep = data2
            nData2 = self.joinImages(data2.numpy())
            # print (data.shape)

            # print("nData shape: ", nData.shape)

            if self.gpu_mode:
                data = nData.cuda()
                #print (data.shape)
                data2 = RGBtoL(nData2).cuda()



            # print ("self.sample_z_.shape", self.sample_z_.shape)
            # for j in range(1, self.class_num):
            #    self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

            temp = torch.zeros((self.class_num, 1))
            for i in range(self.class_num):
                temp[i, 0] = i
            #print(temp)

            temp_y = torch.zeros((self.class_num , 1))
            #print ("tempy", temp_y)
            for i in range(len(temp)):
                temp_y[i] = int(i % self.class_num)
            # print ("tempy", temp_y)
            sample_y_ = torch.zeros((self.class_num , self.class_num)).scatter_(1, temp_y.type(
                torch.LongTensor), 1)

            #print ("self.sample_y_", sample_y_)
            #print ("self.sample_y_", sample_y_[0].unsqueeze(0))

            if self.gpu_mode:
                sample_y_ = sample_y_.cuda()

            if fix:
                samples, G_dep = self.G(sample_y_, data, data2)
            else:
                sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1,
                                                                                  torch.randint(0, self.class_num - 1, (
                                                                                      self.batch_size, 1)).type(
                                                                                      torch.LongTensor), 1)
                sample_z_ = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

                samples, G_dep = self.G(sample_y_, data, data2)


            G_im_outGan = samples
            G_dep_outGan = G_dep

            if self.gpu_mode:
                samples = samples.cpu().data.numpy().transpose(0, 2, 3,
                                                               1)  # los valores son la posicion, corre lo segudno a lo ultimo
            else:
                samples = samples.data.numpy().transpose(0, 2, 3, 1)



            ## TRATO DE PASAR IMAGEN ORIGINAL
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

            G_dep = torch.tensor(G_dep.tolist())
            G_dep = LtoRGB(G_dep)
            G_dep = G_dep.type(torch.FloatTensor)
            G_dep = G_dep.cpu().data.numpy().transpose(0, 2, 3, 1)
            G_dep = np.array(G_dep)

            joined = np.concatenate((realIm, dep))

            if self.wiggle:
                im_aux, im_dep_aux = G_im_outGan, G_dep_outGan
                for i in range(0, samples.shape[0]):
                    index = i
                    for j in range(0, self.wiggleDepth):

                        #print(i,j)


                        if (j == 0 and i == 1):
                            #para tomar el original
                            im_aux, im_dep_aux = G_im_outGan, G_dep_outGan
                            joined = np.concatenate((joined, samples))
                        elif(i == 1):
                            #por el problema de las iteraciones proximas
                            index = 0

                        #imagen generada
                        x = im_aux[index].cpu().data.numpy()
                        x = np.expand_dims(x, axis=0)
                        x = torch.tensor(x.tolist()).cuda()

                        x_dep = im_dep_aux[index].cpu().data.numpy()
                        x_dep = np.expand_dims(x_dep, axis=0)
                        x_dep = torch.tensor(x_dep.tolist()).cuda()

                        y = sample_y_[i].unsqueeze(0)

                        im_aux, im_dep_aux = self.G(y, x, x_dep)

                        joined = np.concatenate((joined, im_aux.cpu().data.numpy().transpose(0, 2, 3, 1)))
            else:
                joined = np.concatenate((joined, samples))
                joined = np.concatenate((joined, G_dep))
            # sadadas
            if newSample is None:
                newSample = joined
            else:
                newSample = np.concatenate((newSample, joined))

        """""

        visual.plot(epoch, newSample, int(len(newSample) /self.cantImages))
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
        torch.save(self.D.state_dict(),
                   os.path.join(save_dir, self.model_name + '_' + self.seed + '_' + epoch + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history_ '+self.seed+'.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_' + self.seed_load + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_' + self.seed_load + '_D.pkl')))

    def wiggleEf(self):
        seed, epoch = self.seed_load.split('_')
        self.visWiggle = utils.VisdomImagePlotter(env_name='Cobo_depth_wiggle_' + seed)
        self.visualize_results(epoch=epoch, dataprint=self.dataprint_test, visual=self.visWiggle)

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