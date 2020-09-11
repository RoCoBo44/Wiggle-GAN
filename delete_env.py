import argparse
import os
import visdom

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--env', type=str, default="Cobo")
    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    return args


args = parse_args()
viz = visdom.Visdom()
viz.delete_env(args.env)

