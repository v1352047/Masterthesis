# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.imgnet import load_imgnet
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_imgnet(flatten=False)


#データの選別
x_train_latter = x_train[5000: ]
x_train = x_train[0: 5000]

t_train_latter = t_train[5000: ]
t_train = t_train[0: 5000]


network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=50,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("first-learn.pkl")
print("Saved Network Parameters!")
