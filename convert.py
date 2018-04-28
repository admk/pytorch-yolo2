import sys

import torch

from darknet import Darknet


def convert(cfgfile, weightfile, outfile):
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    torch.save(m.state_dict(), outfile)
    print('Succesfully saved converted weights to {}.'.format(outfile))


if len(sys.argv) == 4:
    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    outfile = sys.argv[3]
    convert(cfgfile, weightfile, outfile)
else:
    print('Usage: ')
    print('  python convert.py cfgfile weightfile outfile')
