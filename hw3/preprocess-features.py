import base64
import csv
import itertools
import os
import subprocess
import sys

csv.field_size_limit(sys.maxsize)

import h5py
import numpy as np
from tqdm import tqdm

import config


def main():
    FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

    features_shape = (
        82783 + 40504,  # number of images in trainval
        config.output_features,
        config.output_size,
    )
    boxes_shape = (
        features_shape[0],
        4,
        config.output_size,
    )
    if not os.path.exists(config.bottom_up_trainval_path):
        os.chdir('data')
        os.chmod('./download.sh', 0o777)
        # subprocess.run('./download.sh')
        subprocess.run('./download.sh', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir('..')
    path = config.preprocessed_trainval_path

    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    with h5py.File(path, libver='latest') as fd:
        features = fd.create_dataset('features', shape=features_shape, dtype='float32')
        boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
        coco_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='int32')
        widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
        heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')

        readers = []

        path = config.bottom_up_trainval_path

        for entry in os.scandir(path):
            if entry.name.endswith('.tsv'):
                file = open(entry.path, 'r')
                reader = csv.DictReader(file, delimiter='\t', fieldnames=FIELDNAMES)
                readers.append(reader)


        reader = itertools.chain.from_iterable(readers)
        for i, item in enumerate(tqdm(reader, total=features_shape[0])):
            coco_ids[i] = int(item['image_id'])
            widths[i] = int(item['image_w'])
            heights[i] = int(item['image_h'])

            buf = base64.decodebytes(item['features'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, config.output_features)).transpose()
            features[i, :, :array.shape[1]] = array

            buf = base64.decodebytes(item['boxes'].encode('utf8'))
            array = np.frombuffer(buf, dtype='float32')
            array = array.reshape((-1, 4)).transpose()
            boxes[i, :, :array.shape[1]] = array


if __name__ == '__main__':
    main()
