import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用 TensorFlow 的日誌輸出

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CHUNK = 512
RATE = 44100
BUFFER_SECONDS = 0.2

"""
為音訊數據加入隨機噪音，作為數據增強的一部分。
:參數 data: ndarray, 音訊數據
:參數 noise_level: float, 噪音強度，預設為 0.005
:回傳: ndarray, 加入噪音後的音訊數據
:來源: ChatGPT
"""
def add_noise(data, noise_level=0.005):
    noise = np.random.randn(len(data))
    return data + noise_level * noise

"""
從 WAV 音訊文件中提取 MFCC 特徵。
:參數 file_path: str, 音訊檔案路徑
:參數 sr: int, 取樣率，預設為 22050
:參數 n_mfcc: int, 要提取的 MFCC 特徵數量，預設為 20
:參數 target_length: int, 目標樣本長度，預設為 0.2 秒
:參數 augment: bool, 是否進行數據增強，預設為 True
:回傳: ndarray 或 None, MFCC 特徵矩陣，出錯時返回 None
:來源: ChatGPT
"""
def extract_mfcc_matrix(file_path, sr=22050, n_mfcc=20, target_length=int(RATE * BUFFER_SECONDS), augment=True):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.fix_length(y, size=int(sr * BUFFER_SECONDS))

        if augment:
            y = add_noise(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        max_len = 44
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    except Exception as e:
        return None

"""
讀取多個鍵位的音訊樣本，並提取 MFCC 特徵。

:參數 key_list: list, 鍵位列表
:回傳: tuple(ndarray, ndarray), 特徵資料與標籤資料
"""
def get_features(key_list):
    x_train = []
    y_train = []
    for index, key in enumerate(key_list):
        for j in range(1, 11):
            path = f'samples/{key}_{j}.wav'
            mfcc = extract_mfcc_matrix(path, augment=False)
            if mfcc is not None:
                x_train.append(mfcc)
                y_train.append(index)

            mfcc_aug = extract_mfcc_matrix(path, augment=True)
            if mfcc_aug is not None:
                x_train.append(mfcc_aug)
                y_train.append(index)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train

"""
訓練 CNN 模型來識別音訊的特徵。
:參數 x_train: ndarray, 訓練用特徵資料
:參數 y_train: ndarray, 訓練用標籤資料
:參數 num_classes: int, 類別總數
:回傳: keras.Model, 訓練好的模型
"""
def train(x_train, y_train, num_classes):
    print(x_train, y_train)
    if x_train.size == 0 or y_train.size == 0:
        raise ValueError("訓練數據為空，請檢查樣本是否正確生成。")

    x_train = (x_train - np.mean(x_train)) / np.std(x_train)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

    CNN = keras.Sequential(name='Improved_CNN')

    CNN.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=x_train.shape[1:]))
    CNN.add(layers.BatchNormalization())
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Dropout(0.3))

    CNN.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    CNN.add(layers.BatchNormalization())
    CNN.add(layers.MaxPooling2D((2, 2)))
    CNN.add(layers.Dropout(0.3))

    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(64, activation='relu'))
    CNN.add(layers.Dropout(0.5))
    CNN.add(layers.Dense(num_classes, activation='softmax'))

    CNN.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    CNN.fit(x_train, y_train, epochs=50, batch_size=8, validation_split=0.2)
    return CNN

"""
根據鍵字符號列表訓練模型，並儲存訓練好的模型。
:參數 key_list: list, 鍵字符號的列表
:回傳: None
"""
def train_model(key_list):
    x_train, y_train = get_features(key_list)
    if x_train.size == 0 or y_train.size == 0:
        print("訓練數據為空，請檢查樣本是否正確生成。")
        return

    model = train(x_train, y_train, num_classes=len(key_list))
    model.save('model.keras')  # 更新儲存格式為 .keras
