#coding=utf-8
'''images -->lmdb'''
import os
import glob
import argparse
import random
from create_dataset import *

def get_args():
    '''get args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', default='data/lmdb/train')
    parser.add_argument('--data_path', default='data/train')
    parser.add_argument('--label_file', default=None)
    parser.add_argument('--suffix', default='*.jpg')
    return parser.parse_args()

def main(args):
    '''main'''
    if not os.path.exists(args.lmdb_path):
        os.makedirs(args.lmdb_path)
    imgpathlist = []
    labellist = []
    if args.label_file is None:
        filelist = glob.glob(os.path.join(args.data_path, args.suffix))
        random.shuffle(filelist)
        for filename in filelist:
            linepart = filename.split('.')[0].split('_')
            if len(linepart) < 2:
                continue
            imgpathlist.append(filename)
            labellist.append(linepart[-1])
            print filename, linepart[-1]
    else:
        labelfile_list = open(args.label_file, 'r').readlines()
        random.shuffle(labelfile_list)
        for line in labelfile_list:
            line = line.strip()
            arr = line.split(' ')
            if len(arr) < 2:
                continue
            filename = os.path.join(args.data_path, arr[0])
            label = ' '.join(arr[1:])
            print filename, label
            imgpathlist.append(filename)
            labellist.append(label)
    createDataset(args.lmdb_path, imgpathlist, labellist)

if __name__ == '__main__':
    main(get_args())
