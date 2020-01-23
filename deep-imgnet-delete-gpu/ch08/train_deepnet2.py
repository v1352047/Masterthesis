# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from common import config
from common.util import to_cpu, to_gpu
from dataset.imgnet import load_imgnet
from deep_convnet import DeepConvNet
from common.trainer import Trainer


#GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
#===============================================
config.GPU = True
#===============================================


(x_train, t_train), (x_test, t_test) = load_imgnet(flatten=False)



#GPUのメモリにデータを移動
x_train = to_gpu(x_train)
t_train = to_gpu(t_train)
x_test = to_gpu(x_test)
t_test = to_gpu(t_test)



with open('wrong_list.pkl', 'rb') as f:
    wrong_list = pickle.load(f)




#データの選別
x_train_latter = x_train[5000: ]
x_train = x_train[0: 5000]

t_train_latter = t_train[5000: ]
t_train = t_train[0: 5000]


#間違えたデータを除く
x_train = [value for key, value in enumerate(x_train) if not(key in wrong_list)]
t_train = [value for key, value in enumerate(t_train) if not(key in wrong_list)]


#このリストをndarrayに変換
x_train = cp.array(x_train)
t_train = cp.array(t_train)



network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=50,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("second-learn.pkl")
print("Saved Network Parameters!")
