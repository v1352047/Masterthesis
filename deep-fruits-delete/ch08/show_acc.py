# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.imgnet import load_imgnet
from deep_convnet import DeepConvNet


def get_data():
    (x_train, t_train), (x_test, t_test) = load_imgnet(flatten=False, one_hot_label=False)
    return x_test, t_test


x, t = get_data()
network = DeepConvNet()
network.load_params(file_name="first_learn.pkl")
accuracy_cnt = 0


y = network.predict(x[:100])
p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
accuracy_cnt = sum(p == t)


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
print("counter:" + str(len(x)))
