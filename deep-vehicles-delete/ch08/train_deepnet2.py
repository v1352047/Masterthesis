# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.vehicles import load_vehicles
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_vehicles(flatten=False)

with open('wrong-list.pkl', 'rb') as f:
    wrong_list = pickle.load(f)


#データの選別
x_train = x_train[5000: ]
t_train = t_train[5000: ]



#間違えたデータを除く
x_train = [value for key, value in enumerate(x_train) if not(key in wrong_list)]
t_train = [value for key, value in enumerate(t_train) if not(key in wrong_list)]


#このリストをndarrayに変換
x_train = np.array(x_train)
t_train = np.array(t_train)



network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("second-learn.pkl")
print("Saved Network Parameters!")
