"""
Author: Chenhongyi Yang
Reference: We are sorry that we cannot find this script's original authors, but we are appreciate about their work.
"""

import glob
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import cv2
import lmdb

import argparse
parser = argparse.ArgumentParser('Convert LMDB dataset')
parser.add_argument('train-img-dir', 'Path to ImageNet training images')
parser.add_argument('train-out', 'Path to output training lmdb dataset')
parser.add_argument('val-img-dir', 'Path to ImageNet validation images')
parser.add_argument('val-out', 'Path to output validation lmdb dataset')
args = parser.parse_args()

_10TB = 10 * (1 << 40)

class LmdbDataExporter(object):
    """
    making LMDB database
    """
    label_pattern = re.compile(r'/.*/.*?(\d+)$')

    def __init__(self,
                 img_dir=None,
                 output_path=None,
                 batch_size=100):
        """
            img_dir: imgs directory
            output_path: LMDB output path
        """
        self.img_dir = img_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.label_list = list()

        if not os.path.exists(img_dir):
            raise Exception(f'{img_dir} is not exists!')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.lmdb_env = lmdb.open(output_path, map_size=_10TB, max_dbs=4)
        self.label_dict = defaultdict(int)

    def export(self):
        idx = 0
        results = []
        st = time.time()
        iter_img_lst = self.read_imgs()
        length = self.get_length()
        while True:
            items = []
            try:
                while len(items) < self.batch_size:
                    items.append(next(iter_img_lst))
            except StopIteration:
                break

            with ThreadPoolExecutor() as executor:
                results.extend(executor.map(self._extract_once, items))

            if len(results) >= self.batch_size:
                self.save_to_lmdb(results)
                idx += self.batch_size
                et = time.time()
                print(f'time: {(et-st)}(s)  count: {idx}')
                st = time.time()
                if length - idx <= self.batch_size:
                    self.batch_size = 1
                del results[:]

        et = time.time()
        print(f'time: {(et-st)}(s)  count: {idx}')
        self.save_to_lmdb(results)
        self.save_total(idx)
        print('Total length:', len(results))
        del results[:]

    def save_to_lmdb(self, results):
        """
        persist to lmdb
        """
        with self.lmdb_env.begin(write=True) as txn:
            while results:
                img_key, img_byte = results.pop()
                if img_key is None or img_byte is None:
                    continue
                txn.put(img_key, img_byte)

    def save_total(self, total: int):
        """
        persist all numbers of imgs
        """
        with self.lmdb_env.begin(write=True, buffers=True) as txn:
            txn.put('total'.encode(), str(total).encode())

    def _extract_once(self, item) -> Tuple[bytes, bytes]:
        full_path = item[-1]
        imageKey = item[1]

        img = cv2.imread(full_path)
        if img is None:
            print(f'{full_path} is a bad img file.')
            return None, None
        _, img_byte = cv2.imencode('.JPEG', img)
        return (imageKey.encode('ascii'), img_byte.tobytes())

    def get_length(self):
        img_list = glob.glob(os.path.join(self.img_dir, '*/*.JPEG'))
        return len(img_list)

    def read_imgs(self):
        img_list = glob.glob(os.path.join(self.img_dir, '*/*.JPEG'))

        for idx, item_img in enumerate(img_list):
            write_key = os.path.split(item_img)[-1]
            item = (idx, write_key, item_img)
            yield item


if __name__ == '__main__':
    train_input_dir = args.train_img_dir
    train_output_path = args.train_out

    val_input_dir = args.val_img_dir
    val_output_path = args.val_out

    exporter_train = LmdbDataExporter(
        train_input_dir,
        train_output_path,
        batch_size=10000)
    exporter_train.export()

    exporter_val = LmdbDataExporter(
        val_input_dir,
        val_output_path,
        batch_size=10000)
    exporter_val.export()