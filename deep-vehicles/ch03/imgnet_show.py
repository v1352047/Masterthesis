# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.imgnet import load_imgnet
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img), mode="RGB")
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_imgnet(flatten=False, normalize=False)


img = x_train[0]
label = t_train[0]
print(label)  # apple, orange, banana or melon

print(img.shape)  # (784,)
img = img.reshape(100, 100, -1)  # 形状を元の画像サイズに変形
print(img.shape)  # (28, 28)

img_show(img)
