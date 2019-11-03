from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import torch
from PIL import Image
import numpy as np

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
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
        cams = ImagesDataset(root_dir=os.getcwd() + '\images')
        data_loader = DataLoader(cams , batch_size=batch_size, shuffle=True)

    return data_loader


class ImagesDataset(Dataset):
    """My dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.nCameras = 2

    def __len__(self):

        oneCameRoot = self.root_dir + '\CAM1'
        return int(len([name for name in os.listdir(oneCameRoot) if os.path.isfile(os.path.join(oneCameRoot, name))])/2) #por el depth

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        #folders por camaras / los llamo CAM + nCameras
        #con el root llego a antes de que se abran las ramas y desde ahi agarro cada foto con el Index que quiero, deph y normal
        #sample = {}
        sample = np.array([])
        for i in range(0,self.nCameras):
            oneCameRoot = self.root_dir + '\CAM' + str(i)

            #foto normal
            img_name = os.path.join(oneCameRoot, str(idx).zfill(4)+ "_n.png")
            img = Image.open(img_name).convert('L')
            data = np.array(img)
            data = np.true_divide(data, [255.0], out=None)
            data = (data * 2) - 1

            #foto produndidad
            img_name = os.path.join(oneCameRoot, str(idx).zfill(4) + "_d.png")
            img = Image.open(img_name).convert('L') #el LA es para blanco y negro
            data2 = np.array(img)
            data2 = np.true_divide(data2, [255.0], out=None)
            data2 = (data2 * 2) - 1 # Para que quede entre -1 y 1

            """ #Para que se guarde la imagen
            
            data3 = ((data + 1) / 2 ) * 255.0
            outIm = Image.fromarray(data3)
            if outIm.mode != 'RGB':
                outIm = outIm.convert('RGB')
            outIm.save(self.root_dir + '\CAM' + "im.png")
            """

            s = np.array([data,data2])

            if sample.size == 0:
                sample = s
            else:
                sample = np.concatenate([sample,s])
            ##Esto lo trata como un diccionario
            #sample['Cam' + str(i) + "_d"] = data
            #sample['Cam' + str(i) + "_n"] = data2
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __iter__(self):

        for i in range(this.__len__()):
            list.append(this.__getitem__(i))
        return iter(list)

"""
im = ImagesDataset(root_dir=os.getcwd() + '\images')  #os.getcwd()
print(im.__len__())
ima = im.__getitem__(idx=1)
imagen = im.__getitem__(idx=1)
print(0)
print(imagen.shape)
print(1)
print(imagen['Cam1_n'].shape[1])
"""