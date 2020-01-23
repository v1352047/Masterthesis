# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np



dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/imgnet.pkl"

train_num = 2800
test_num = 800
img_dim = (3, 100, 100)
img_size = 30000

       
def _change_one_hot_label(X):
    T = np.zeros((X.size, 4))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_imgnet(normalize=True, flatten=True, one_hot_label=False):
    """ImageNetデータセットの読み込み
    
    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label : 
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか 
    
    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])
    
    if not flatten:
         for key in ('train_img', 'test_img'):
             dataset[key] = dataset[key].reshape(-1, 3, 100, 100)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


