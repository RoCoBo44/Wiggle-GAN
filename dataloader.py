from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
from configparser import ConfigParser
import matplotlib.pyplot as plt
import os
import torch as th
from PIL import Image
import numpy as np
import random
from PIL import ImageMath
import random

def dataloader(dataset, input_size, batch_size,dim,split='train', trans=False):
    #transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
    #                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == '4cam':
        if split != 'test':
            cams = ImagesDataset(root_dir=os.getcwd() + '/Images/ActualDataset', dim=dim, name=split, transform=trans)
            return DataLoader(cams, batch_size=batch_size, shuffle=True, num_workers=3)
        else:
            cams = TestingDataset(root_dir=os.getcwd() + '/Images/Input-Test', dim=dim, name=split)
            return DataLoader(cams, batch_size=batch_size, shuffle=False, num_workers=3)

    return data_loader


class ImagesDataset(Dataset):
    """My dataset."""

    def __init__(self, root_dir, dim, name, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.nCameras = 2
        self.imageDim = dim
        self.name = name
        self.parser = ConfigParser()
        self.parser.read('config.ini')
        self.transform = transform

    def __len__(self):

        return self.parser.getint(self.name, 'total')
        #oneCameRoot = self.root_dir + '\CAM1'
        #return int(len([name for name in os.listdir(oneCameRoot) if os.path.isfile(os.path.join(oneCameRoot, name))])/2) #por el depth


    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        idx = self.parser.get(self.name, str(idx))
        if self.transform:
            brighness = random.uniform(0.7, 1.2)
            saturation = random.uniform(0, 2)
            contrast = random.uniform(0.4, 2)
            gamma = random.uniform(0.7, 1.3)
            hue = random.uniform(-0.3, 0.3)  # 0.01
        """""
        sample = np.array([])
        for i in range(0,self.nCameras):
            oneCameRoot = self.root_dir + '/CAM' + str(i)

            #foto normal
            img_name = os.path.join(oneCameRoot, "n_" + idx+ ".png")
            img = Image.open(img_name).convert('RGB')#.convert('L')
            if (img.size[0] != self.imageDim or img.size[1] !=self.imageDim):
                img = img.resize((self.imageDim,self.imageDim))

            if self.transform:
                img = transforms.functional.adjust_gamma(img, gamma)
                img = transforms.functional.adjust_brightness(img, brighness)
                img = transforms.functional.adjust_contrast(img, contrast)
                img = transforms.functional.adjust_saturation(img, saturation)
                img = transforms.functional.adjust_hue(img, hue)
                #img.show()

            data = np.array(img)
            #print(data.shape)
            data = (data/ 255.0)
            data = data.transpose(2, 0, 1)
            #print(data.shape)

            #bright_tform = Grayscale(keep_channels=True)
            #t_data = bright_tform(th.from_numpy(data))
            #t_data = t_data.numpy()





            #t_data2 = t_data.transpose(1, 2, 0)
            #print(t_data2.shape)
            #t_data2 = t_data2 * 255.0
            #t_data2 = t_data2.astype(np.uint8)
            #print(t_data)
            #outIm = Image.fromarray(t_data2, mode='RGB')
            #outIm.show()


            ## H W C


            #foto produndidad
            img_name = os.path.join(oneCameRoot, "d_" + idx + ".png")
            #print (Image.open(img_name).mode)
            img = Image.open(img_name).convert('I')#.convert('RGB')#.convert('L') #el LA es para blanco y negro
            img = convert_I_to_L(img).convert('RGB')# para 3 canales
            if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
                img = img.resize((self.imageDim, self.imageDim))
            data2 = np.array(img)
            #print (data2)
            data2 = np.true_divide(data2, [255.0], out=None)
            #print (data2.shape)
            #data2 = np.expand_dims(data2, axis=2)
            data2 = data2.transpose(2, 0, 1)

            #show_image(data2, grey=False)

            #Para que se guarde la imagen
            #data3 = ((data + 1.0) / 2.0 ) * 255.0
            #data3 = data3.transpose(1, 2, 0)
            #data3 = data3.astype(np.uint8)  ## int != uint8
            #print(data3)
            #print(data3.shape)
            #outIm = Image.fromarray(data3,mode='RGB')
            #outIm.show()
            #outIm.save(self.root_dir + '\CAM' + "im.png")
            #stop
           # print (data.shape)
            s = np.array([data,data2])
           # print (s.shape)

            if sample.size == 0:
                sample = s
            else:
                sample = np.concatenate([sample,s])
        """""

        oneCameRoot = self.root_dir + '/CAM0'

        # foto normal
        img_name = os.path.join(oneCameRoot, "n_" + idx + ".png")
        img = Image.open(img_name).convert('RGB')  # .convert('L')
        if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
            img = img.resize((self.imageDim, self.imageDim))
        if self.transform:
            img = transforms.functional.adjust_gamma(img, gamma)
            img = transforms.functional.adjust_brightness(img, brighness)
            img = transforms.functional.adjust_contrast(img, contrast)
            img = transforms.functional.adjust_saturation(img, saturation)
            img = transforms.functional.adjust_hue(img, hue)
        x1 = transforms.ToTensor()(img)
        x1 = (x1 * 2) - 1

        # foto produndidad
        img_name = os.path.join(oneCameRoot, "d_" + idx + ".png")
        img = Image.open(img_name).convert('I')
        img = convert_I_to_L(img)
        if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
            img = img.resize((self.imageDim, self.imageDim))
        x1_dep = transforms.ToTensor()(img)
        x1_dep = (x1_dep * 2) - 1

        oneCameRoot = self.root_dir + '/CAM1'

        # foto normal
        img_name = os.path.join(oneCameRoot, "n_" + idx + ".png")
        img = Image.open(img_name).convert('RGB')  # .convert('L')
        if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
            img = img.resize((self.imageDim, self.imageDim))
        if self.transform:
            img = transforms.functional.adjust_gamma(img, gamma)
            img = transforms.functional.adjust_brightness(img, brighness)
            img = transforms.functional.adjust_contrast(img, contrast)
            img = transforms.functional.adjust_saturation(img, saturation)
            img = transforms.functional.adjust_hue(img, hue)
        x2 = transforms.ToTensor()(img)
        x2 = (x2 * 2) - 1

        # foto produndidad
        img_name = os.path.join(oneCameRoot, "d_" + idx + ".png")
        img = Image.open(img_name).convert('I')
        img = convert_I_to_L(img)
        if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
            img = img.resize((self.imageDim, self.imageDim))
        x2_dep = transforms.ToTensor()(img)
        x2_dep = (x2_dep * 2) - 1


        #random izq o derecha
        if (bool(random.getrandbits(1))):
            sample = {'x_im': x1, 'x_dep': x1_dep, 'y_im': x2, 'y_dep': x2_dep, 'y_': torch.tensor(1.)}
        else:
            sample = {'x_im': x2, 'x_dep': x2_dep, 'y_im': x1, 'y_dep': x1_dep, 'y_': torch.tensor(0.)}

        return sample

    def __iter__(self):

        for i in range(this.__len__()):
            list.append(this.__getitem__(i))
        return iter(list)

class TestingDataset(Dataset):
    """My dataset."""

    def __init__(self, root_dir, dim, name):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.imageDim = dim
        self.name = name
        files = os.listdir(self.root_dir)
        self.files = [ele for ele in files if not ele.endswith('_d.png')]

    def __len__(self):

        #return self.parser.getint(self.name, 'total')
        #oneCameRoot = self.root_dir + '\CAM1'
        #return int(len([name for name in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, name))])/2) #por el depth
        return len(self.files)


    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        # foto normal
        img_name = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(img_name).convert('RGB')  # .convert('L')
        if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
            img = img.resize((self.imageDim, self.imageDim))
        x1 = transforms.ToTensor()(img)
        x1 = (x1 * 2) - 1


        # foto produndidad
        img_name = os.path.join(self.root_dir , self.files[idx][:-4] + "_d.png")
        img = Image.open(img_name).convert('I')
        img = convert_I_to_L(img)
        if (img.size[0] != self.imageDim or img.size[1] != self.imageDim):
            img = img.resize((self.imageDim, self.imageDim))
        x1_dep = transforms.ToTensor()(img)
        x1_dep = (x1_dep * 2) - 1

        sample = {'x_im': x1, 'x_dep': x1_dep}

        return sample

    def __iter__(self):

        for i in range(this.__len__()):
            list.append(this.__getitem__(i))
        return iter(list)


def show_image(t_data, grey=False):

    #from numpy
    t_data2 = t_data.transpose(1, 2, 0)
    t_data2 = t_data2 * 255.0
    t_data2 = t_data2.astype(np.uint8)
    if (not grey):
        outIm = Image.fromarray(t_data2, mode='RGB')
    else:
        t_data2 = np.squeeze(t_data2, axis=2)
        outIm = Image.fromarray(t_data2, mode='L')
    outIm.show()

def convert_I_to_L(img):
    array = np.uint8(np.array(img) / 256) #el numero esta bien, sino genera espacios en negro en la imagen
    return Image.fromarray(array)
