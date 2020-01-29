# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append("../../../dataset")  # 親ディレクトリのファイルをインポートするための設定
from common import config
from common.util import to_cpu, to_gpu
from fruits import load_fruits
from deep_convnet import DeepConvNet
from common.trainer import Trainer


#GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
#===============================================
config.GPU = True
#===============================================


(x_train, t_train), (x_test, t_test) = load_fruits(flatten=False)


#データの選別
x_train = x_train[: 5000]
t_train = t_train[: 5000]


#GPUのメモリにデータを移動
#x_train = to_gpu(x_train)
#t_train = to_gpu(t_train)
#x_test = to_gpu(x_test)
#t_test = to_gpu(t_test)




network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=40,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=100)
trainer.train()

# パラメータの保存
network.save_params("first-learn.pkl")
print("Saved Network Parameters!")
