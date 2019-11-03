import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.cuda as cu
import torch.optim as optim
from PIL import Image
from dataloader import dataloader
from torch.autograd import Variable

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=4, output_dim=1, input_shape=3, class_num=10, zdim=1, height = 10, width = 10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        print ("self.output_dim", self.output_dim)
        self.class_num = class_num
        self.input_shape = list(input_shape)
        self.zdim = zdim
        self.toPreDecov = 1024
        self.toDecov = 1
        self.height = height
        self.width = width

        self.input_shape.insert(1,2) #esto cambio despues por colores
        print("input shpe gen",self.input_shape)

        self.conv1 = nn.Sequential(
            ##############RED SUPER CHICA PARA QUE ANDE TO DO PORQUE RAM Y MEMORY
            nn.Conv2d(self.input_dim, 2, 4, 2, 1), #para mi el 2 tendria que ser 1
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, 2, 4, 2, 1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, 1, 4, 2, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        )

        self.n_size = self._get_conv_output(self.input_shape)
        self.cubic = (self.n_size // 8192)
        print("self.cubic: ",self.cubic)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size, self.toPreDecov),
            nn.BatchNorm1d(self.toPreDecov),
            nn.LeakyReLU(0.2),
        )

        self.preDeconv = nn.Sequential(
            ##############RED SUPER CHICA PARA QUE ANDE TO DO PORQUE RAM Y MEMORY

            #nn.Linear(self.toPreDecov + self.zdim + self.class_num, 1024),
            #nn.BatchNorm1d(1024),
            #nn.LeakyReLU(0.2),
            #nn.Linear(1024, self.toDecov * self.height // 64  * self.width// 64),
            #nn.BatchNorm1d(self.toDecov * self.height // 64  * self.width// 64),
            #nn.LeakyReLU(0.2),
            #nn.Linear(self.toDecov * self.height // 64 * self.width // 64 , self.toDecov * self.height // 32 * self.width // 32),
            #nn.BatchNorm1d(self.toDecov * self.height // 32 * self.width // 32),
            #nn.LeakyReLU(0.2),
            #nn.Linear(self.toDecov * self.height // 32 * self.width // 32,
            #         1 * self.height * self.width),
            #nn.BatchNorm1d(1 * self.height * self.width),
            #nn.LeakyReLU(0.2),

            nn.Linear(self.toPreDecov + self.zdim + self.class_num, 60),
            nn.BatchNorm1d(60),
            nn.LeakyReLU(0.2),
            nn.Linear(60, 1 * self.height * self.width),
            nn.BatchNorm1d(1 * self.height * self.width),
            nn.Tanh(), #Cambio porque hago como que termino ahi

        )

        """
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.toDecov, 2, 4, 2, 0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.ConvTranspose2d(2, self.output_dim, 4, 2, 1),
            nn.Tanh(), #esta recomendado que la ultima sea TanH de la Generadora da valores entre -1 y 1
        )
        """
        utils.initialize_weights(self)


    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        print("inShape:",input.shape)
        output_feat = self.conv1(input.squeeze())
        print ("output_feat",output_feat.shape)
        n_size = output_feat.data.view(bs, -1).size(1)
        print ("n",n_size // 4)
        return n_size // 4

    def forward(self, input, clase, im, imDep):

        ##Esto es lo que voy a hacer
        # Cat entre la imagen y la profundidad
        print ("H",self.height,"W",self.width)
        imDep = imDep[:, None, :, :]
        im = im[:, None, :, :]
        print ("imdep",imDep.shape)
        print ("im",im.shape)
        x = torch.cat([im, imDep], 1)

        #Ref Conv de ese cat
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        #print ("x:",x.shape)

        #cat entre el ruido y la clase
        y = torch.cat([input, clase], 1)
        print("Cat entre input y clase", y.shape) #podria separarlo, unir primero con clase y despues con ruido

        #Red Lineal que une la Conv con el cat anterior
        x = torch.cat([x, y], 1)
        x = self.preDeconv(x)
        print ("antes de deconv", x.shape)
        x = x.view(-1, self.toDecov, self.height, self.width)
        print("Despues View: ", x.shape)
        #Red que saca produce la imagen final
        #x = self.deconv(x)
        print("La salida de la generadora es: ",x.shape)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_shape =2, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_shape = list(input_shape)
        self.class_num = class_num

        self.input_shape.insert(1,2) #esto cambio despues por colores
        print(self.input_shape)

        """""
          in_channels (int): Number of channels in the input image
          out_channels (int): Number of channels produced by the convolution
          kernel_size (int or tuple): Size of the convolving kernel -  lo que se agarra para la conv
          stride (int or tuple, optional): Stride of the convolution. Default: 1
          padding (int or tuple, optional): Zero-padding added to both sides of the input.
          """""

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1), #para mi el 2 tendria que ser 1
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4, stride=2),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(4, stride=2),
            nn.Conv2d(32, 20, 4, 2, 1),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2),
        )

        self.n_size = self._get_conv_output(self.input_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size // 4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Softmax(dim=1),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
            nn.Softmax(dim=1),
        )
        utils.initialize_weights(self)

        # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input.squeeze())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, input, im):

        #esto va a cambiar cuando tenga color
        if (len(input.shape) <= 3):
            input = input[:, None, :, :]
        im = im[:, None, :, :]
        print(input.shape)
        print(im.shape)
        x = torch.cat([input, im], 1)
        print(input.shape)
        #print("this si X:", x)
        #print("now shape", x.shape)
        x = x.type(torch.FloatTensor)
        x = x.to(device='cuda:0')
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c

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
        self.z_dim = 62
        self.class_num = (args.cameras - 1) * 2 #un calculo que hice en paint
        self.sample_num = self.class_num ** 2


        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.batch_size = self.batch_size * self.nCameras * (self.nCameras - 1) ## EXPLICADO EN VIDEO es por los frames
        data = self.data_loader.__iter__().__next__()[0]
        print ("Aca va la data")
        print(data)

        # networks init
        print("Values de entrada a las redes")
        print("data.shape[0]", data.shape[0])
        print("data.shape[1]", data.shape[1])
        print(data.shape)
        ## estoy muy perdido de como seria la entrada, para mi seria el anchox el alto de cada imagen pero tampoco se como la red se conecta Y alto y acho
        self.G = generator(input_dim=2, output_dim=1, input_shape=data.shape, class_num=self.class_num,zdim = self.z_dim, height = data.shape[1], width = data.shape[2])
        #Ese 2 del input es porque es blanco y negro (imINICIO+imANGULO)
        self.D = discriminator(input_dim=2, output_dim=1, input_shape=data.shape, class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

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


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("self.batch_size",self.batch_size)
        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()
        print("y_real_",self.y_real_)
        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter,data in enumerate(self.data_loader): #Cambiar con el dataset, agarra por batch size

                #Aumento mi data
                x_im,x_dep,y_im,y_ = self.rearrengeData(data)
                #para que usen Tensor
                x_im  = torch.tensor(x_im.tolist())
                y_im  = torch.tensor(y_im.tolist())
                x_dep = torch.tensor(x_dep.tolist())
                y_    = torch.tensor(y_.tolist())

                x_im  = x_im.type(torch.FloatTensor)
                y_im  = y_im.type(torch.FloatTensor)
                x_dep = x_dep.type(torch.FloatTensor)
                y_    = y_.type(torch.FloatTensor)
                # x_im  = imagenes normales
                # x_dep = profundidad de images
                # y_im  = imagen con el angulo cambiado
                # y_    = angulo de la imagen = tengo que tratar negativos

                print(y_.shape)
                print(y_im.shape)
                print(x_im.shape)
                print(x_dep.shape)
                y_ = y_ + (self.nCameras -1)
                y_ = torch.Tensor(list(map(lambda x: int(x-1) if (x > 0)else int(x), y_)))
                #x_im  = torch.Tensor(list(x_im))
                #x_dep = torch.Tensor(x_dep)
                #y_im  = torch.Tensor(y_im)
                print(y_.shape[0])
                if iter == self.data_loader.dataset.__len__() * self.nCameras * (self.nCameras - 1)// self.batch_size:
                    print ("Break")
                    break
                z_ = torch.rand((y_.shape[0], self.z_dim))
                print (y_.type(torch.LongTensor).unsqueeze(1))
                y_vec_ = torch.zeros((y_.shape[0], self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)

                print("y_vec_",y_vec_)
                print ("z_",z_)

                if self.gpu_mode:
                    x_im, z_, y_vec_, y_im,x_dep = x_im.cuda(), z_.cuda(), y_vec_.cuda(),y_im.cuda(),x_dep.cuda()
                # update D network
                self.D_optimizer.zero_grad()

                # data[0] normal 0 camaraActual
                # data[1] dep
                # data[2] normal 1 camaraComparada
                # data[3] dep

                D_real, C_real = self.D(y_im, x_im)  ## Es la funcion forward `` g(z) x
                #   print(D_real)   ##no se bien que es pero como se compara con el numero real, o es que tan seguro esta que es el resultado o tendria que dar el resultado
                #   print(len(D_real))
                #   print(C_real) ##El vector que dice que tanto piensa que es el numero de salida (por eso se compara con el Y_vec que ya tiene el resultado)

                print ("D_real",D_real)
                print ("self.y_real_", self.y_real_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_, x_im, x_dep)
                print ("im shape",x_im.shape )
                print("G shape", G_.shape)
                D_fake, C_fake = self.D(G_,x_im)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_, x_im, x_dep)
                D_fake, C_fake = self.D(G_, x_im)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()
                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                           D_loss.item(), G_loss.item()))

                """""""""
                for camaraComparada in range(self.class_num+1):
                    for camaraActual in range(self.class_num+1):

                        if camaraActual == camaraComparada:
                            print("equal")
                        else:
                            z_ = torch.rand((self.batch_size, self.z_dim))

                            y_vec_ = torch.zeros(cantMov)
                            ubicacion = (camaraComparada - camaraActual) + self.class_num

                            y_vec_ = torch.zeros((self.batch_size, cantMov)).scatter_(1, ubicacion(torch.LongTensor).unsqueeze(1), 1)
                            print ("vector",y_vec_)

                            # update D network
                            self.D_optimizer.zero_grad()
                            
                            #data[0] normal 0 camaraActual
                            #data[1] dep
                            #data[2] normal 1 camaraComparada
                            #data[3] dep
                            

                            D_real, C_real = self.D(data[camaraActual * 2], data[camaraComparada * 2])  ## Es la funcion forward
                            #   print(D_real)   ##no se bien que es pero como se compara con el numero real, o es que tan seguro esta que es el resultado o tendria que dar el resultado
                            #   print(len(D_real))
                            #   print(C_real) ##El vector que dice que tanto piensa que es el numero de salida (por eso se compara con el Y_vec que ya tiene el resultado)
                            D_real_loss = self.BCE_loss(D_real, self.y_real_)
                            C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                            G_ = self.G(z_, y_vec_, data[camaraActual * 2], data[camaraActual * 2 + 1])
                            D_fake, C_fake = self.D(G_,  data[camaraComparada * 2])
                            D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                            C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                            D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                            self.train_hist['D_loss'].append(D_loss.item())

                            D_loss.backward()
                            self.D_optimizer.step()

                            # update G network
                            self.G_optimizer.zero_grad()

                            G_ = self.G(z_, y_vec_, data[camaraActual * 2], data[camaraActual * 2 + 1])
                            D_fake, C_fake = self.D(G_, data[camaraComparada * 2])

                            G_loss = self.BCE_loss(D_fake, self.y_real_)
                            C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                            G_loss += C_fake_loss
                            self.train_hist['G_loss'].append(G_loss.item())

                            G_loss.backward()
                            self.G_optimizer.step()
                            if ((iter + 1) % 100) == 0:
                                print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                                      ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                                       D_loss.item(), G_loss.item()))
                                       
                #   print(x_) #Supongo que imagenes para labels pero cargadas con -1, solo lo deduzco porque ambas (y_) son de la misma cantidad. tiene 4 dim
             #   print(len(x_))
             #   print(y_)  #LABELS
             #   print(len(y_))
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
             #   print(y_vec_) #es un conjunto de vectores en donde tiene el valor 1 en el lugar del label correspondiente

                if self.gpu_mode:
                    x_, z_, y_vec_ = x_.cuda(), z_.cuda(), y_vec_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

             #   print(x_)
                D_real, C_real = self.D(x_) ## Es la funcion forward
             #   print(D_real)   ##no se bien que es pero como se compara con el numero real, o es que tan seguro esta que es el resultado o tendria que dar el resultado
             #   print(len(D_real))
             #   print(C_real) ##El vector que dice que tanto piensa que es el numero de salida (por eso se compara con el Y_vec que ya tiene el resultado)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()
                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))
            """""""""
            print ("Sali del loop")
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

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

        cantidadIm = 1
        image_frame_dim = int(np.floor(np.sqrt(cantidadIm * self.class_num)))
        root = os.getcwd() + '\images' + '\TEST'
        idx = 0
        img_name = os.path.join(root, str(idx).zfill(4) + "_n.png")
        img = Image.open(img_name).convert('L')
        data = np.array(img)
        data = np.true_divide(data, [255.0], out=None)
        data = (data * 2) - 1
        print ("normal image :",data.shape)

        nData = self.joinImages(data)

        img_name = os.path.join(root, str(idx).zfill(4) + "_d.png")
        img = Image.open(img_name).convert('L')  # el LA es para blanco y negro
        data2 = np.array(img)
        data2 = np.true_divide(data2, [255.0], out=None)
        data2 = (data2 * 2) - 1
        print ("depth image :", data2.shape)

        nData2 = self.joinImages(data2)

        print("nData shape: ", nData.shape)

        if self.gpu_mode:
            data, data2 = nData.cuda(), nData2.cuda()


        self.sample_z_ = torch.rand(cantidadIm * self.class_num, self.z_dim)
        print ("self.sample_z_.shape", self.sample_z_.shape)
            #for j in range(1, self.class_num):
            #    self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i
        print(temp)

        temp_y = torch.zeros((self.class_num * cantidadIm, 1))
        print ("tempy", temp_y)
        for i in range(len(temp)):
            temp_y[i] = int(i % self.class_num)
        print ("tempy", temp_y)
        self.sample_y_ = torch.zeros((self.class_num * cantidadIm, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)

        print ("self.sample_y_",self.sample_y_)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_, self.sample_y_, data, data2)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_, data, data2)
        print("normal")
        print(samples.shape)

        root_dir = os.getcwd() + '\images'

        """
        data3 = ((samples + 1) / 2) * 255.0
        print ("getting one",data3[0,:,:,:].shape)
        data3 = data3.cpu().numpy()
        outIm = Image.fromarray(data3[0,:,:,:].squeeze())
        if outIm.mode != 'RGB':
            outIm = outIm.convert('RGB')
        outIm.save(root_dir + '\CAM' + "im.png")
        """
        
        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1) #los valores son la posicion, corre lo segudno a lo ultimo
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)
        print("Trans")
        print(samples)
        samples = (samples + 1) / 2
        print(" normaliza ")
        print(samples)

        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def joinImages(self, data):
        nData = []
        for i in range(self.class_num):
            nData.append(data)
        nData = np.array(nData)
        nData = torch.tensor(nData.tolist())
        nData = nData.type(torch.FloatTensor)

        return nData

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

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
