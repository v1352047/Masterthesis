# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append("../../../dataset")  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from fruits import load_fruits
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_fruits(flatten=True, normalize=False)

img = x_train[10]
label = t_train[10]
print(label)
img = img.reshape(100, 100, -1)
pil_img = Image.fromarray(np.uint8(img), mode="RGB")
pil_img.show()
exit(0)


network = DeepConvNet()  
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=20, mini_batch_size=20,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("first-learn.pkl")
print("Saved Network Parameters!")
