# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common import config
from common.util import to_cpu, to_gpu
import cupy as cp
import pickle
from dataset.imgnet import load_imgnet
from common.functions import sigmoid, softmax




#GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
#===============================================
config.GPU = True
#===============================================




def get_data():
    (x_train, t_train), (x_test, t_test) = load_imgnet(normalize=True, one_hot_label=False)
    

    #GPUのメモリにデータを移動
    x_train = to_gpu(x_train)
    t_train = to_gpu(t_train)

    
    #データの選別
    x_train_former = x_train[0: 5000]
    x_train = x_train[5000: ]

    t_train_former = t_train[0: 5000]
    t_train = t_train[5000: ]

    return x_train, t_train


def init_network():
    with open("first-learn.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2 = network['b1'], network['b2']

    a1 = cp.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = cp.dot(z1, W2) + b2
    y = softmax(a2)

    return y



x, t = get_data()
network = init_network()
accuracy_cnt = 0
wrong_ans = []


for i in range(len(x)):
    y = predict(network, x[i])
    p= cp.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1
    else:
        wrong_ans.append(i)


#間違えた問題のindexを保存
with open('wrong_list.pkl', 'wb') as f:
    pickle.dump(wrong_ans, f)


print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
print("counter:" + str(len(x)))
