import os
import sys
import torch
import pickle


class Logger(object):
    """ Logger class. """
    def __init__(self, path=None):
        self.console = sys.stdout
        self.file = None
        if path is not None:
            self.file = open(path, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
                self.file.close()


class AverageMeter(object):
    """ Compute and store the average and current value. """
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()  # Reset the values.

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count


def save_checkpoint(state, path='./checkpoint.pth'):
    """ Save current state as checkpoint. """
    torch.save(state, path)


def load_checkpoint(net, opt, path='./checkpoint.pth'):
    """ Load previous pre-trained checkpoint.
    :param net:  Network instance.
    :param opt:  Optimizer instance.
    :param path: Path of checkpoint file.
    :return:     Checkpoint epoch number.
    """
    if os.path.isfile(path):
        print('=> Loading checkpoint {}...'.format(path))
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['opt'])
        return checkpoint['epoch']
    else:
        raise ValueError('=> No checkpoint found at {}.'.format(path))


def load_vgg16_caffe(net, path='./5stage-vgg.py36pickle'):
    """ Load network parameters from VGG-16 Caffe model. """
    load_pretrained_caffe(net, path, only_vgg=True)


def load_pretrained_caffe(net, path='./hed_pretrained_bsds.py36pickle', only_vgg=False):
    """ Load network parameters from pre-trained HED Caffe model. """
    # Read pretrained parameters.
    with open(path, 'rb') as f:
        pretrained_params = pickle.load(f)

    # Load parameters into network.
    print('=> Start loading parameters...')
    vgg_layers_name = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    for name, param in net.named_parameters():
        _, layer_name, var_name = name.split('.')
        if (only_vgg is False) or ((only_vgg is True) and (layer_name in vgg_layers_name)):
            param.data.copy_(torch.from_numpy(pretrained_params[layer_name][var_name]))
            print('=> Loaded {}.'.format(name))
    print('=> Finish loading parameters.')
