import torch.nn as nn
import utils, torch
from torch.autograd import Variable
import torch.nn.functional as F


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=4, output_dim=1, input_shape=3, class_num=10, height=10, width=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # print ("self.output_dim", self.output_dim)
        self.class_num = class_num
        self.input_shape = list(input_shape)
        self.toPreDecov = 1024
        self.toDecov = 1
        self.height = height
        self.width = width

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores

        # print("input shpe gen",self.input_shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 10, 4, 2, 1),  # para mi el 2 tendria que ser 1
            nn.Conv2d(10, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 3, 4, 2, 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2),
        )

        self.n_size = self._get_conv_output(self.input_shape)
        # print ("self.n_size",self.n_size)
        self.cubic = (self.n_size // 8192)
        # print("self.cubic: ",self.cubic)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size, self.n_size),
            nn.BatchNorm1d(self.n_size),
            nn.LeakyReLU(0.2),
        )

        self.preDeconv = nn.Sequential(
            ##############RED SUPER CHICA PARA QUE ANDE TO DO PORQUE RAM Y MEMORY

            # nn.Linear(self.toPreDecov + self.zdim + self.class_num, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, self.toDecov * self.height // 64  * self.width// 64),
            # nn.BatchNorm1d(self.toDecov * self.height // 64  * self.width// 64),
            # nn.LeakyReLU(0.2),
            # nn.Linear(self.toDecov * self.height // 64 * self.width // 64 , self.toDecov * self.height // 32 * self.width // 32),
            # nn.BatchNorm1d(self.toDecov * self.height // 32 * self.width // 32),
            # nn.LeakyReLU(0.2),
            # nn.Linear(self.toDecov * self.height // 32 * self.width // 32,
            #         1 * self.height * self.width),
            # nn.BatchNorm1d(1 * self.height * self.width),
            # nn.LeakyReLU(0.2),

            nn.Linear(self.n_size + self.class_num, 400),
            nn.BatchNorm1d(400),
            nn.LeakyReLU(0.2),
            nn.Linear(400, 800),
            nn.BatchNorm1d(800),
            nn.LeakyReLU(0.2),
            nn.Linear(800, self.output_dim * self.height * self.width),
            nn.BatchNorm1d(self.output_dim * self.height * self.width),
            nn.Tanh(),  # Cambio porque hago como que termino ahi

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
        # print("inShape:",input.shape)
        output_feat = self.conv1(input.squeeze())
        # print ("output_feat",output_feat.shape)
        n_size = output_feat.data.view(bs, -1).size(1)
        # print ("n",n_size // 4)
        return n_size // 4

    def forward(self, clase, im):
        ##Esto es lo que voy a hacer
        # Cat entre la imagen y la profundidad
        # print ("H",self.height,"W",self.width)
        # imDep = imDep[:, None, :, :]
        # im = im[:, None, :, :]
        x = im

        # Ref Conv de ese cat
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # print ("x:", x.shape)
        x = self.fc1(x)
        # print ("x:",x.shape)

        # cat entre el ruido y la clase
        y = clase
        # print("Cat entre input y clase", y.shape) #podria separarlo, unir primero con clase y despues con ruido

        # Red Lineal que une la Conv con el cat anterior
        x = torch.cat([x, y], 1)
        x = self.preDeconv(x)
        # print ("antes de deconv", x.shape)
        x = x.view(-1, self.output_dim, self.height, self.width)
        # print("Despues View: ", x.shape)
        # Red que saca produce la imagen final
        # x = self.deconv(x)
        # print("La salida de la generadora es: ",x.shape)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_shape=2, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim * 2  # ya que le doy el origen
        self.output_dim = output_dim
        self.input_shape = list(input_shape)
        self.class_num = class_num

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores
        # print(self.input_shape)

        """""
          in_channels (int): Number of channels in the input image
          out_channels (int): Number of channels produced by the convolution
          kernel_size (int or tuple): Size of the convolving kernel -  lo que se agarra para la conv
          stride (int or tuple, optional): Stride of the convolution. Default: 1
          padding (int or tuple, optional): Zero-padding added to both sides of the input.
          """""

        """
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
          """

        self.conv = nn.Sequential(

            nn.Conv2d(self.input_dim, 4, 4, 2, 1),  # para mi el 2 tendria que ser 1
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),

        )

        self.n_size = self._get_conv_output(self.input_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size // 4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(64, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(64, self.class_num),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

        # generate input sample and forward to get shape

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input.squeeze())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, input, origen):
        # esto va a cambiar cuando tenga color
        # if (len(input.shape) <= 3):
        #    input = input[:, None, :, :]
        # im = im[:, None, :, :]
        # print("D in shape",input.shape)

        # print(input.shape)
        # print("this si X:", x)
        # print("now shape", x.shape)
        x = input
        x = x.type(torch.FloatTensor)
        x = x.to(device='cuda:0')

        x = torch.cat((x, origen), 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c


#######################################################################################################################
class UnetConvBlock(nn.Module):
    '''
    Convolutional block of a U-Net:
    Conv2d - Batch normalization - LeakyReLU
    Conv2D - Batch normalization - LeakyReLU
    Basic Dropout (optional)
    '''

    def __init__(self, in_size, out_size, dropout=0.0, stride=1):
        '''
        Constructor of the convolutional block
        '''
        super(UnetConvBlock, self).__init__()

        # Convolutional layer with IN_SIZE --> OUT_SIZE
        conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1,
                          padding=1)  # podria aplicar stride 2
        # Activation unit
        activ_unit1 = nn.LeakyReLU(0.2)
        # Add batch normalization if necessary
        self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_size), activ_unit1)

        # Convolutional layer with OUT_SIZE --> OUT_SIZE
        conv2 = nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3, stride=stride,
                          padding=1)  # podria aplicar stride 2
        # Activation unit
        activ_unit2 = nn.LeakyReLU(0.2)

        # Add batch normalization
        self.conv2 = nn.Sequential(conv2, nn.BatchNorm2d(out_size), activ_unit2)

        # Dropout
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, inputs):
        '''
        Do a forward pass
        '''
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        if not (self.drop is None):
            outputs = self.drop(outputs)
        return outputs


class UnetDeSingleConvBlock(nn.Module):
    '''
    DeConvolutional block of a U-Net:
    Conv2d - Batch normalization - LeakyReLU
    Basic Dropout (optional)
    '''

    def __init__(self, in_size, out_size, dropout=0.0, stride=1, padding=1):
        '''
        Constructor of the convolutional block
        '''
        super(UnetDeSingleConvBlock, self).__init__()

        # Convolutional layer with IN_SIZE --> OUT_SIZE
        conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=stride, padding=1)
        # Activation unit
        activ_unit1 = nn.LeakyReLU(0.2)
        # Add batch normalization if necessary
        self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_size), activ_unit1)

        # Dropout
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, inputs):
        '''
        Do a forward pass
        '''
        outputs = self.conv1(inputs)
        if not (self.drop is None):
            outputs = self.drop(outputs)
        return outputs


class UnetDeconvBlock(nn.Module):
    '''
    DeConvolutional block of a U-Net:
    UnetDeSingleConvBlock (skip_connection)
    Cat last_layer + skip_connection
    UnetDeSingleConvBlock ( Cat )
    Basic Dropout (optional)
    '''

    def __init__(self, in_size_layer, in_size_skip_con, out_size, dropout=0.0):
        '''
        Constructor of the convolutional block
        '''
        super(UnetDeconvBlock, self).__init__()

        self.conv1 = UnetDeSingleConvBlock(in_size_skip_con, in_size_skip_con, dropout)
        self.conv2 = UnetDeSingleConvBlock(in_size_skip_con + in_size_skip_con, out_size, dropout)

        # Dropout
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, inputs_layer, inputs_skip):
        '''
        Do a forward pass
        '''

        outputs = self.conv1(inputs_skip)

        outputs = changeDim(outputs, inputs_layer)

        outputs = torch.cat((inputs_layer, outputs), 1)
        outputs = self.conv2(outputs)

        return outputs


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_size_layer, in_size_skip_con, out_size, bilinear=True):
        super(UpBlock, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_size_layer // 2, in_size_layer // 2, kernel_size=2, stride=2)

        self.conv = UnetDeconvBlock(in_size_layer, in_size_skip_con, out_size)

    def forward(self, inputs_layer, inputs_skip):

        inputs_layer = self.up(inputs_layer)

        # input is CHW
        inputs_layer = changeDim(inputs_layer, inputs_skip)

        return self.conv(inputs_layer, inputs_skip)


class lastBlock(nn.Module):
    '''
    DeConvolutional block of a U-Net:
    Conv2d - Batch normalization - LeakyReLU
    Basic Dropout (optional)
    '''

    def __init__(self, in_size, out_size, dropout=0.0):
        '''
        Constructor of the convolutional block
        '''
        super(lastBlock, self).__init__()

        # Convolutional layer with IN_SIZE --> OUT_SIZE
        conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=1, padding=1)
        # Activation unit
        activ_unit1 = nn.Tanh()
        # Add batch normalization if necessary
        self.conv1 = nn.Sequential(conv1, nn.BatchNorm2d(out_size), activ_unit1)

        # Dropout
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

    def forward(self, inputs):
        '''
        Do a forward pass
        '''
        outputs = self.conv1(inputs)
        if not (self.drop is None):
            outputs = self.drop(outputs)
        return outputs


################

class generator_UNet(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=4, output_dim=1, input_shape=3, class_num=2, expand_net=3):
        super(generator_UNet, self).__init__()
        self.input_dim = input_dim + 1  # por la clase
        self.output_dim = output_dim
        # print ("self.output_dim", self.output_dim)
        self.class_num = class_num
        self.input_shape = list(input_shape)

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores

        self.expandNet = expand_net  # 5

        # Downsampling
        self.conv1 = UnetConvBlock(self.input_dim, pow(2, self.expandNet), stride=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = UnetConvBlock(pow(2, self.expandNet), pow(2, self.expandNet + 1), stride=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = UnetConvBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 2), stride=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # Middle ground
        self.conv4 = UnetDeSingleConvBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), stride=2)
        # UpSampling
        self.up1 = UpBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), pow(2, self.expandNet + 1),
                           bilinear=True)
        self.up2 = UpBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 1), pow(2, self.expandNet),
                           bilinear=True)
        self.up3 = UpBlock(pow(2, self.expandNet), pow(2, self.expandNet), 8, bilinear=True)
        self.last = lastBlock(8, self.output_dim)

        utils.initialize_weights(self)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        # print("inShape:",input.shape)
        output_feat = self.conv1(input.squeeze())  ##CAMBIAR
        # print ("output_feat",output_feat.shape)
        n_size = output_feat.data.view(bs, -1).size(1)
        # print ("n",n_size // 4)
        return n_size // 4

    def forward(self, clase, im):
        x = im

        ##PARA TENER LA CLASE DEL CORRIMIENTO
        cl = ((clase == 1))
        cl = cl[:, 1]
        cl = cl.type(torch.FloatTensor)
        max = (clase.size())[1] - 1
        cl = cl / float(max)

        ##crear imagen layer de corrimiento
        tam = im.size()
        layerClase = torch.ones([tam[0], tam[2], tam[3]], dtype=torch.float32, device="cuda:0")
        for idx, item in enumerate(layerClase):
            layerClase[idx] = item * cl[idx]
        layerClase = layerClase.unsqueeze(0)
        layerClase = layerClase.transpose(1, 0)

        ##unir layer el rgb de la imagen
        x = torch.cat((x, layerClase), 1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)  # self.maxpool1(x1))
        x3 = self.conv3(x2)  # self.maxpool2(x2))
        x4 = self.conv4(x3)  # self.maxpool3(x3))
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = changeDim(x, im)
        x = self.last(x)

        return x


class discriminator_UNet(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_shape=[2, 2], class_num=10, expand_net = 2):
        super(discriminator_UNet, self).__init__()
        self.input_dim = input_dim * 2  # ya que le doy el origen
        self.output_dim = output_dim
        self.input_shape = list(input_shape)
        self.class_num = class_num

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores

        self.expandNet = expand_net  # 4

        # Downsampling
        self.conv1 = UnetConvBlock(self.input_dim, pow(2, self.expandNet), stride=1, dropout=0.3)
        self.conv2 = UnetConvBlock(pow(2, self.expandNet), pow(2, self.expandNet + 1), stride=2, dropout=0.5)
        self.conv3 = UnetConvBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 2), stride=2, dropout=0.4)

        # Middle ground
        self.conv4 = UnetDeSingleConvBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), stride=2,
                                           dropout=0.3)

        self.n_size = self._get_conv_output(self.input_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size // 4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )

        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            # nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
            nn.Softmax(dim=1),  # poner el que la suma da 1
        )
        utils.initialize_weights(self)

        # generate input sample and forward to get shape

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        x = input.squeeze()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        n_size = x.data.view(bs, -1).size(1)
        return n_size

    def forward(self, input, origen):
        # esto va a cambiar cuando tenga color
        # if (len(input.shape) <= 3):
        #    input = input[:, None, :, :]
        # im = im[:, None, :, :]
        # print("D in shape",input.shape)

        # print(input.shape)
        # print("this si X:", x)
        # print("now shape", x.shape)
        x = input
        x = x.type(torch.FloatTensor)
        x = x.to(device='cuda:0')

        x = torch.cat((x, origen), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c


def changeDim(x, y):
    ''' Change dim-image from x to y '''

    diffY = torch.tensor([y.size()[2] - x.size()[2]])
    diffX = torch.tensor([y.size()[3] - x.size()[3]])
    x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                  diffY // 2, diffY - diffY // 2])
    return x


########################################      ACGAN        ###########################################################

class depth_generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=4, output_dim=1, input_shape=3, class_num=10, zdim=1, height=10, width=10):
        super(depth_generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.class_num = class_num
        # print ("self.output_dim", self.output_dim)
        self.input_shape = list(input_shape)
        self.zdim = zdim
        self.toPreDecov = 1024
        self.toDecov = 1
        self.height = height
        self.width = width

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores

        # print("input shpe gen",self.input_shape)

        self.conv1 = nn.Sequential(
            ##############RED SUPER CHICA PARA QUE ANDE TO DO PORQUE RAM Y MEMORY
            nn.Conv2d(self.input_dim, 2, 4, 2, 1),  # para mi el 2 tendria que ser 1
            nn.Conv2d(2, 1, 4, 2, 1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2),
        )

        self.n_size = self._get_conv_output(self.input_shape)
        # print ("self.n_size",self.n_size)
        self.cubic = (self.n_size // 8192)
        # print("self.cubic: ",self.cubic)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size, self.n_size),
            nn.BatchNorm1d(self.n_size),
            nn.LeakyReLU(0.2),
        )

        self.preDeconv = nn.Sequential(
            ##############RED SUPER CHICA PARA QUE ANDE TO DO PORQUE RAM Y MEMORY

            # nn.Linear(self.toPreDecov + self.zdim + self.class_num, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, self.toDecov * self.height // 64  * self.width// 64),
            # nn.BatchNorm1d(self.toDecov * self.height // 64  * self.width// 64),
            # nn.LeakyReLU(0.2),
            # nn.Linear(self.toDecov * self.height // 64 * self.width // 64 , self.toDecov * self.height // 32 * self.width // 32),
            # nn.BatchNorm1d(self.toDecov * self.height // 32 * self.width // 32),
            # nn.LeakyReLU(0.2),
            # nn.Linear(self.toDecov * self.height // 32 * self.width // 32,
            #         1 * self.height * self.width),
            # nn.BatchNorm1d(1 * self.height * self.width),
            # nn.LeakyReLU(0.2),

            nn.Linear(self.n_size + self.zdim + self.class_num, 50),
            nn.BatchNorm1d(50),
            nn.LeakyReLU(0.2),
            nn.Linear(50, 200),
            nn.BatchNorm1d(200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, self.output_dim * self.height * self.width),
            nn.BatchNorm1d(self.output_dim * self.height * self.width),
            nn.Tanh(),  # Cambio porque hago como que termino ahi

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
        # print("inShape:",input.shape)
        output_feat = self.conv1(input.squeeze())
        # print ("output_feat",output_feat.shape)
        n_size = output_feat.data.view(bs, -1).size(1)
        # print ("n",n_size // 4)
        return n_size // 4

    def forward(self, input, clase, im, imDep):
        ##Esto es lo que voy a hacer
        # Cat entre la imagen y la profundidad
        print ("H", self.height, "W", self.width)
        # imDep = imDep[:, None, :, :]
        # im = im[:, None, :, :]
        print ("imdep", imDep.shape)
        print ("im", im.shape)
        x = torch.cat([im, imDep], 1)

        # Ref Conv de ese cat
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        print ("x:", x.shape)
        x = self.fc1(x)
        # print ("x:",x.shape)

        # cat entre el ruido y la clase
        y = torch.cat([input, clase], 1)
        print("Cat entre input y clase", y.shape)  # podria separarlo, unir primero con clase y despues con ruido

        # Red Lineal que une la Conv con el cat anterior
        x = torch.cat([x, y], 1)
        x = self.preDeconv(x)
        print ("antes de deconv", x.shape)
        x = x.view(-1, self.output_dim, self.height, self.width)
        print("Despues View: ", x.shape)
        # Red que saca produce la imagen final
        # x = self.deconv(x)
        print("La salida de la generadora es: ", x.shape)

        return x


class depth_discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_shape=2, class_num=10):
        super(depth_discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_shape = list(input_shape)
        self.class_num = class_num

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores
        print(self.input_shape)

        """""
          in_channels (int): Number of channels in the input image
          out_channels (int): Number of channels produced by the convolution
          kernel_size (int or tuple): Size of the convolving kernel -  lo que se agarra para la conv
          stride (int or tuple, optional): Stride of the convolution. Default: 1
          padding (int or tuple, optional): Zero-padding added to both sides of the input.
          """""

        """
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
          """

        self.conv = nn.Sequential(

            nn.Conv2d(self.input_dim, 4, 4, 2, 1),  # para mi el 2 tendria que ser 1
            nn.LeakyReLU(0.2),
            nn.Conv2d(4, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, 4, 2, 1),
            nn.BatchNorm2d(16),

        )

        self.n_size = self._get_conv_output(self.input_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(self.n_size // 4, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(64, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(64, self.class_num),
            nn.Sigmoid(),
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
        # esto va a cambiar cuando tenga color
        # if (len(input.shape) <= 3):
        #    input = input[:, None, :, :]
        # im = im[:, None, :, :]
        print("D in shape", input.shape)
        print("D im shape", im.shape)
        x = torch.cat([input, im], 1)
        print(input.shape)
        # print("this si X:", x)
        # print("now shape", x.shape)
        x = x.type(torch.FloatTensor)
        x = x.to(device='cuda:0')
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c


class depth_generator_UNet(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=4, output_dim=1, input_shape=3, class_num=10, zdim=1, expand_net=3):
        super(depth_generator_UNet, self).__init__()
        self.input_dim = input_dim + 1
        self.output_dim = output_dim  # por depth +1 (al final no)
        self.class_num = class_num
        # print ("self.output_dim", self.output_dim)
        self.input_shape = list(input_shape)
        self.zdim = zdim

        self.expandNet = expand_net  # 5

        self.input_shape[1] = self.input_dim  # esto cambio despues por colores

        # Downsampling
        self.conv1 = UnetConvBlock(self.input_dim, pow(2, self.expandNet))
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = UnetConvBlock(pow(2, self.expandNet), pow(2, self.expandNet + 1), stride=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = UnetConvBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 2), stride=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        # Middle ground
        self.conv4 = UnetDeSingleConvBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), stride=2)
        # UpSampling
        self.up1 = UpBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), pow(2, self.expandNet + 1))
        self.up2 = UpBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 1), pow(2, self.expandNet))
        self.up3 = UpBlock(pow(2, self.expandNet), pow(2, self.expandNet), 8)
        self.last = lastBlock(8, self.output_dim)

        self.upDep1 = UpBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), pow(2, self.expandNet + 1))
        self.upDep2 = UpBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 1), pow(2, self.expandNet))
        self.upDep3 = UpBlock(pow(2, self.expandNet), pow(2, self.expandNet), 8)
        self.lastDep = lastBlock(8, 1)

        self.n_size = self._get_conv_output(self.input_shape)
        self.cubic = (self.n_size // 8192)

        utils.initialize_weights(self)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv1(input.squeeze())  ##CAMBIAR
        n_size = output_feat.data.view(bs, -1).size(1)
        # print ("n",n_size // 4)
        return n_size // 4

    def forward(self, z, clase, im, imDep):
        ##Hago algo con el z?
        #print (im.shape)
        #print (z.shape)
        #print (z)
        #imz = torch.repeat_interleave(z, repeats=torch.tensor([2, 2]), dim=1)
        #print (imz.shape)
        #print (imz)
        #sdadsadas

        x = torch.cat([im, imDep], 1)

        ##PARA TENER LA CLASE DEL CORRIMIENTO
        cl = ((clase == 1))
        cl = cl[:, 1]
        cl = cl.type(torch.FloatTensor)
        max = (clase.size())[1] - 1
        cl = cl / float(max)

        ##crear imagen layer de corrimiento
        tam = im.size()
        layerClase = torch.ones([tam[0], tam[2], tam[3]], dtype=torch.float32, device="cuda:0")
        for idx, item in enumerate(layerClase):
            layerClase[idx] = item * cl[idx]
        layerClase = layerClase.unsqueeze(0)
        layerClase = layerClase.transpose(1, 0)

        ##unir layer el rgb de la imagen
        x = torch.cat((x, layerClase), 1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)  # self.maxpool1(x1))
        x3 = self.conv3(x2)  # self.maxpool2(x2))
        x4 = self.conv4(x3)  # self.maxpool3(x3))

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        #x = changeDim(x, im)
        x = self.last(x)

        dep = self.upDep1(x4, x3)
        dep = self.upDep2(dep, x2)
        dep = self.upDep3(dep, x1)
        # x = changeDim(x, im)
        dep = self.lastDep(dep)

        #print(x.shape)
        #dep = x[:,3,:,:].squeeze().unsqueeze(0).transpose(0,1)
        x = x[:, :3, :, :]

        return x, dep


class depth_discriminator_UNet(discriminator_UNet):
    def __init__(self, input_dim=1, output_dim=1, input_shape=[2, 2], class_num=10, expand_net=2):
        discriminator_UNet.__init__(self, input_dim=input_dim, output_dim=output_dim, input_shape=input_shape,
                                    class_num=class_num, expand_net = expand_net)

        self.input_dim = input_dim * 2 + 1 # ya que le doy el origen + mapa de profundidad
        self.conv1 = UnetConvBlock(self.input_dim, pow(2, self.expandNet), stride=1, dropout=0.3)
        self.conv2 = UnetConvBlock(pow(2, self.expandNet), pow(2, self.expandNet + 1), stride=2, dropout=0.2)
        self.conv3 = UnetConvBlock(pow(2, self.expandNet + 1), pow(2, self.expandNet + 2), stride=2, dropout=0.2)
        self.conv4 = UnetDeSingleConvBlock(pow(2, self.expandNet + 2), pow(2, self.expandNet + 2), stride=2,
                                           dropout=0.3)

        self.input_shape[1] = self.input_dim
        self.n_size = self._get_conv_output(self.input_shape)

        utils.initialize_weights(self)

    def forward(self, input, origen, dep):
        # esto va a cambiar cuando tenga color
        # if (len(input.shape) <= 3):
        #    input = input[:, None, :, :]
        # im = im[:, None, :, :]
        # print("D in shape",input.shape)

        # print(input.shape)
        # print("this si X:", x)
        # print("now shape", x.shape)
        x = input
        x = x.type(torch.FloatTensor)
        x = x.to(device='cuda:0')

        x = torch.cat((x, origen), 1)
        x = torch.cat((x, dep), 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c
