import numpy as np
import pickle
import glob
import matplotlib.pylab as plt



#全画像のファイル名を取得
apples = glob.glob('apple-100/*.jpg')
bananas = glob.glob('banana-100/*.jpg')
oranges = glob.glob('orange-100/*.jpg')
melons = glob.glob('melon-100/*.jpg')



#全画像データをリストで取得
for key, value in enumerate(apples):
    apples[key] = plt.imread(value)

for key, value in enumerate(bananas):
    bananas[key] = plt.imread(value)

for key, value in enumerate(oranges):
    oranges[key] = plt.imread(value)

for key, value in enumerate(melons):
    melons[key] = plt.imread(value)



#リストをNumPy配列に変換
apples = np.array(apples)
bananas = np.array(bananas)
oranges = np.array(oranges)
melons = np.array(melons)


#全画像を1次元に整形
apples = apples.reshape(apples.shape[0], 30000)
bananas = bananas.reshape(bananas.shape[0], 30000)
oranges = oranges.reshape(oranges.shape[0], 30000)
melons = melons.reshape(melons.shape[0], 30000)


#pickleに変換するために整形
train_img = np.concatenate([apples[:700], bananas[:700], oranges[:700], melons[:700]])
train_label = np.array(([0] * 700) + ([1] * 700) + ([2] * 700) + ([3] * 700))
test_img = np.concatenate([apples[700: 800], bananas[700: 800], oranges[700: 800], melons[700: 800]])
test_label = np.array(([0] * 100) + ([1] * 100) + ([2] * 100) + ([3] * 100))


#データをシャッフル
shuffle_train = np.random.permutation(np.arange(2800))
shuffle_test = np.random.permutation(np.arange(400))

train_img = np.array([train_img[i] for i in shuffle_train])
train_label = np.array([train_label[i] for i in shuffle_train])
test_img = np.array([test_img[i] for i in shuffle_test])
test_label = np.array([test_label[i] for i in shuffle_test])



#ディクショナリーを作成
dataset = {'train_img': train_img,
           'train_label': train_label,
           'test_img': test_img,
           'test_label': test_label}



#pickleに書き出し
with open('imgnet.pkl', 'wb') as f:
    pickle.dump(dataset, f)



