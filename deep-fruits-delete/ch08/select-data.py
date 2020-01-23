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
x_train_former = x_train[0: 5000]
x_train = x_train[5000: ]

t_train_former = t_train[0: 5000]
t_train = t_train[5000: ]



network = DeepConvNet()  
network.load_params(file_name="first-learn.pkl")
accuracy = 0
wrong_ans = []


y = network.predict(x_test)
p = np.argmax(y, 0)
wrong_ans = [key for key, value in enumerate(p == t_train) if not value]
accuracy = sum(p == t_train) / len(x_train)


#間違えた問題のindexを保存
with open('wrong_list.pkl', 'wb') as f:
    pickle.dump(wrong_ans, f)
