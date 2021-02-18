import os
import os.path as osp

BASE_DIR = osp.dirname(osp.abspath(__file__))

DATA_DIR = osp.join(BASE_DIR, 'data')
LOG_DIR = osp.join(DATA_DIR, 'log')
MODEL_DIR = osp.join(DATA_DIR, 'model')