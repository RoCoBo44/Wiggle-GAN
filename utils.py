import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
import visdom
import random

def load_mnist(dataset):
    data_dir = os.path.join("./data", dataset)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY).astype(np.int)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1

    X = X.transpose(0, 3, 1, 2) / 255.
    # y_vec = y_vec.transpose(0, 3, 1, 2)

    X = torch.from_numpy(X).type(torch.FloatTensor)
    y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    return X, y_vec

def load_celebA(dir, transform, batch_size, shuffle):
    # transform = transforms.Compose([
    #     transforms.CenterCrop(160),
    #     transform.Scale(64)
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])

    # data_dir = 'data/celebA'  # this path depends on your computer
    dset = datasets.ImageFolder(dir, transform)
    data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

    return data_loader


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    #print ("shape", images.shape)
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]== 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            #print("indez ",idx)
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
    images = []
    for e in range(num):
        img_name = path + '_epoch%04d' % (e+1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x1 = range(len(hist['D_loss_train']))
    x2 = range(len(hist['G_loss_train']))

    y1 = hist['D_loss_train']
    y2 = hist['G_loss_train']

    if (x1 != x2):
        y1 = [0.0] * (len(y2) - len(y1)) + y1
        x1 = x2

    plt.plot(x1, y1, label='D_loss_train')

    plt.plot(x2, y2, label='G_loss_train')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
        self.ini = False
    def plot(self, var_name,names, split_name, hist):



        x = []
        y = []
        for i, name in enumerate(names):
            x.append(len(hist[name ]))
            y.append(hist[name])
        #x1 = (len(hist['D_loss_' +split_name]))
        #x2 = (len(hist['G_loss_' +split_name]))

        #y1 = hist['D_loss_'+split_name]
        #y2 = hist['G_loss_'+split_name]

        if (x[0] != x[1]):
            y[0] = [0.0] * (len(y[1]) - len(y[0])) + y[0]
            x[0] = x[1]
        np.array(x)

        for i,n in enumerate(names):
            x[i] = np.arange(1, x[i]+1)


        if not self.ini:
            for i, name in enumerate(names):
                if i == 0:
                    self.win = self.viz.line(X=x[i], Y=np.array(y[i]), env=self.env,name = name,opts=dict(
                        title=var_name + '_'+split_name, showlegend = True
                    ))
                else:
                    self.viz.line(X=x[i], Y=np.array(y[i]), env=self.env,win=self.win, name=name, update='append')
            self.ini = True
        else:
            x[0] = np.array([x[0][-2], x[0][-1]])

            for i,n in enumerate(names):
                y[i] = np.array([y[i][-2], y[i][-1]])
                self.viz.line(X=x[0], Y=np.array(y[i]), env=self.env, win=self.win, name=n, update='append')


class VisdomLineTwoPlotter(VisdomLinePlotter):

    def plot(self, var_name, epoch,names, hist):

        x1 = epoch
        y1 = hist[names[0]]
        y2 = hist[names[1]]
        y3 = hist[names[2]]
        y4 = hist[names[3]]


        #y1 = hist['D_loss_' + split_name]
        #y2 = hist['G_loss_' + split_name]
        #y3 = hist['D_loss_' + split_name2]
        #y4 = hist['G_loss_' + split_name2]


        #x1 = np.arange(1, x1+1)

        if not self.ini:
            self.win = self.viz.line(X=np.array([x1]), Y=np.array(y1), env=self.env,name = names[0],opts=dict(
                title=var_name,
                showlegend = True,
                linecolor = np.array([[0, 0, 255]])
            ))
            self.viz.line(X=np.array([x1]), Y=np.array(y2), env=self.env,win=self.win, name=names[1],
                          update='append', opts=dict(
                    linecolor=np.array([[255, 153, 51]])
                ))
            self.viz.line(X=np.array([x1]), Y=np.array(y3), env=self.env, win=self.win, name=names[2],
                          update='append', opts=dict(
                    linecolor=np.array([[0, 51, 153]])
                ))
            self.viz.line(X=np.array([x1]), Y=np.array(y4), env=self.env, win=self.win, name=names[3],
                          update='append', opts=dict(
                    linecolor=np.array([[204, 51, 0]])
                ))
            self.ini = True
        else:

            y4 = np.array([y4[-2], y4[-1]])
            y3 = np.array([y3[-2], y3[-1]])
            y2 = np.array([y2[-2], y2[-1]])
            y1 = np.array([y1[-2], y1[-1]])
            x1 = np.array([x1 - 1, x1])
            self.viz.line(X=x1, Y=np.array(y1), env=self.env, win=self.win, name=names[0], update='append')
            self.viz.line(X=x1, Y=np.array(y2), env=self.env, win=self.win, name=names[1], update='append')
            self.viz.line(X=x1, Y=np.array(y3), env=self.env, win=self.win, name=names[2],
                          update='append')
            self.viz.line(X=x1, Y=np.array(y4), env=self.env, win=self.win, name=names[3],
                          update='append')

class VisdomImagePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = visdom.Visdom()
        self.env = env_name
    def plot(self, epoch,images,rows):

        list_images = []
        for image in images:
            #print (image)
            list_images.append(image.transpose([2, 0, 1]) * 255)
        self.viz.images(
            list_images,
            padding=2,
            nrow =rows,
            opts=dict(title="epoch: " + str(epoch)),
            env=self.env
        )


def augmentData(x,y, randomness = 1):
    """
    :param x: image X
    :param y: image Y
    :param randomness: Value of randomness (between 1 and 0)
    :return: data x,y augmented
    """


    sampleX = []
    sampleY = []

    #print(type(y))
    #print (y.shape)

    for numpyX, numpyY in zip(x,y):

        # Preparing to get image #
        aumX = numpyX.transpose(1, 2, 0)
        aumY = numpyY.transpose(1, 2, 0)
        aumX = (aumX+1)/2 * 255.0
        aumY = (aumY + 1) / 2 * 255.0

        aumX = aumX.astype(np.uint8)
        aumY = aumY.astype(np.uint8)
        imgX = Image.fromarray(aumX, mode='RGB')
        imgY = Image.fromarray(aumY, mode='RGB')

        # Values for augmentation #
        brighness = random.uniform(0.4, 1.5)* randomness + (1-randomness)
        saturation = random.uniform(0, 2)* randomness + (1-randomness)
        contrast = random.uniform(0.4, 2)* randomness + (1-randomness)
        gamma = random.uniform(0.7, 1.3)* randomness + (1-randomness)
        hue = random.uniform(-0.01, 0.01)* randomness

        imgX = transforms.functional.adjust_gamma(imgX, gamma)
        imgX = transforms.functional.adjust_brightness(imgX, brighness)
        imgX = transforms.functional.adjust_contrast(imgX, contrast)
        imgX = transforms.functional.adjust_saturation(imgX, saturation)
        imgX = transforms.functional.adjust_hue(imgX, hue)
        #imgX.show()

        imgY = transforms.functional.adjust_gamma(imgY, gamma)
        imgY = transforms.functional.adjust_brightness(imgY, brighness)
        imgY = transforms.functional.adjust_contrast(imgY, contrast)
        imgY = transforms.functional.adjust_saturation(imgY, saturation)
        imgY = transforms.functional.adjust_hue(imgY, hue)
        #imgY.show()

        sx = np.array(imgX)
        # print(data.shape)
        sx = ((sx / 255.0)*2)-1
        sx = sx.transpose(2, 0, 1)

        sy = np.array(imgY)
        # print(data.shape)
        sy = ((sy / 255.0)*2)-1
        sy = sy.transpose(2, 0, 1)


        sampleX.append(sx)
        sampleY.append(sy)

    return np.array(sampleX),np.array(sampleY)