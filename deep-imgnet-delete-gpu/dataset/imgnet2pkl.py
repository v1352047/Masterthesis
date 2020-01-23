import numpy as np
import pickle
import glob
import matplotlib.pylab as plt


#全画像のファイル名を取得
apples = glob.glob('apple100f/*.jpg')
bananas = glob.glob('banana100f/*.jpg')
oranges = glob.glob('orange100f/*.jpg')
melons = glob.glob('melon100f/*.jpg')
print("Got all files names.")



#全画像データをリストで取得
for key, value in enumerate(apples):
    apples[key] = plt.imread(value)

for key, value in enumerate(bananas):
    bananas[key] = plt.imread(value)
#    デバッグ用コード
    print(key)
    print(value)
    print(len(bananas[key]))
    print(len(bananas[key][0]))
    print(len(bananas[key][0][0]))

for key, value in enumerate(oranges):
    oranges[key] = plt.imread(value)

for key, value in enumerate(melons):
    melons[key] = plt.imread(value)

print("Got the lists of images.")



#リストをNumPy配列に変換
apples = np.array(apples)
bananas = np.array(bananas)
oranges = np.array(oranges)
melons = np.array(melons)

print("Changed the lists into numpy.")



#全画像を1次元に整形
apples = apples.reshape(apples.shape[0], 30000)
bananas = bananas.reshape(bananas.shape[0], 30000)
oranges = oranges.reshape(oranges.shape[0], 30000)
melons = melons.reshape(melons.shape[0], 30000)

print("Changed the shapes of images lists.")


#pickleに変換するために整形
train_img = np.concatenate([apples[:2000], bananas[:2000], oranges[:2000], melons[:2000]])
train_label = np.array(([0] * 2000) + ([1] * 2000) + ([2] * 2000) + ([3] * 2000))
test_img = np.concatenate([apples[2000: 3200], bananas[2000: 3200], oranges[700: 3200], melons[2000: 3200]])
test_label = np.array(([0] * 1200) + ([1] * 1200) + ([2] * 1200) + ([3] * 1200))

print("Arranged all data to change into pickle.")


#データをシャッフル
shuffle_train = np.random.permutation(np.arange(8000))
shuffle_test = np.random.permutation(np.arange(4800))

print("Shuffled all data.")


train_img = np.array([train_img[i] for i in shuffle_train])
train_label = np.array([train_label[i] for i in shuffle_train])
test_img = np.array([test_img[i] for i in shuffle_test])
test_label = np.array([test_label[i] for i in shuffle_test])




#ディクショナリーを作成
dataset = {'train_img': train_img,
           'train_label': train_label,
           'test_img': test_img,
           'test_label': test_label}

print("Created dictionary.")



#pickleに書き出し
with open('imgnet.pkl', 'wb') as f:
    pickle.dump(dataset, f)


print("All data saved!")

