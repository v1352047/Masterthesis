# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

#
sys.path.append('..')

import numpy as np
import cupy as cp

#
from common import config


from common.util import to_cpu, to_gpu



import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

x_train = to_gpu(x_train)
t_train = to_gpu(t_train)
x_test = to_gpu(x_test)
t_test = to_gpu(t_test)


network = DeepConvNet()

#network.load_params("deep_convnet_params.pkl")

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存, パラメーターはGPU用のもの
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")

# 27 minutes
