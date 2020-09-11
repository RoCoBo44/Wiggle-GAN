import argparse
import os
import torch
from GAN import GAN
from CGAN import CGAN
from LSGAN import LSGAN
from DRAGAN import DRAGAN
from ACGAN import ACGAN
from WGAN import WGAN
from WGAN_GP import WGAN_GP
from infoGAN import infoGAN
from EBGAN import EBGAN
from BEGAN import BEGAN
from MyACGAN import MyACGAN
from MyGAN import MyGAN

"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='MyGAN',
                        choices=['GAN', 'CGAN', 'infoGAN', 'ACGAN', 'EBGAN', 'BEGAN', 'WGAN', 'WGAN_GP', 'DRAGAN',
                                 'LSGAN', 'MyACGAN', 'MyGAN'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='4cam',
                        choices=['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'svhn', 'stl10', 'lsun-bed', '4cam'],
                        help='The name of dataset')
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--epoch', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=10, help='The size of input image')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--cameras', type=int, default=2)
    parser.add_argument('--imageDim', type=int, default=128)
    parser.add_argument('--epochV', type=int, default=20)
    parser.add_argument('--cIm', type=int, default=4)
    parser.add_argument('--seedLoad', type=str, default="0000")
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--zGF', type=float, default=0.2)  # Estos valores estan dados por el paper y lo trabajan con otra red, posiblemente hay que cambiarlos
    parser.add_argument('--zDF', type=float, default=0.2)
    parser.add_argument('--bF', type=float, default=0.2)
    parser.add_argument('--expandGen', type=int, default=3)
    parser.add_argument('--expandDis', type=int, default=3)
    parser.add_argument('--wiggle', type=bool, default=False)
    parser.add_argument('--wiggleDepth', type=int, default=2)

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

        # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'CGAN':
        gan = CGAN(args)
    elif args.gan_type == 'ACGAN':
        gan = ACGAN(args)
    elif args.gan_type == 'infoGAN':
        gan = infoGAN(args, SUPERVISED=False)
    elif args.gan_type == 'EBGAN':
        gan = EBGAN(args)
    elif args.gan_type == 'WGAN':
        gan = WGAN(args)
    elif args.gan_type == 'WGAN_GP':
        gan = WGAN_GP(args)
    elif args.gan_type == 'DRAGAN':
        gan = DRAGAN(args)
    elif args.gan_type == 'LSGAN':
        gan = LSGAN(args)
    elif args.gan_type == 'BEGAN':
        gan = BEGAN(args)
    elif args.gan_type == 'MyACGAN':
        gan = MyACGAN(args)
    elif args.gan_type == 'MyGAN':
        gan = MyGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    # launch the graph in a session
    if (not args.wiggle ):
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        #gan.visualize_results(args.epoch)
        #print(" [*] Testing finished!")

    else:
        gan.wiggleEf()
        print(" [*] Wiggle finished!")


if __name__ == '__main__':
    main()
