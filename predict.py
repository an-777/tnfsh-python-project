import pyaudio
import wave
import os
import time
from pynput import keyboard
import threading
import queue
from cnn import extract_mfcc_matrix
import tensorflow as tf
import numpy as np

# 錄音參數
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
BUFFER_SECONDS = 0.2
BUFFER_SIZE = int(RATE / CHUNK * BUFFER_SECONDS)

# 全域變數
frames = []
p = None
stream = None
stop_flag = False
message_queue = None

"""
開始錄音，並持續錄取音訊直到停止標誌被設置為 True。
:參數 None
:回傳: None
"""
def start_audio_stream():
    global p, stream, stop_flag
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        # message_queue.put("錄音開始...")
        message_queue.put("請按下要預測的鍵...")

        while not stop_flag:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError:
                stream.stop_stream()
                stream.close()
                stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
"""
儲存錄音片段為 WAV 檔案。
:參數 None
:回傳: str, 儲存的 WAV 檔案路徑
"""
def save_sample():
    global frames, CHUNK, RATE
    output_dir = "samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    time.sleep(0.1)

    filename = os.path.join(output_dir, "predict.wav")
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames[-int(RATE / CHUNK * 0.2):]))
    wf.close()

    message_queue.put("錄音完成，已儲存為 predict.wav")
    return filename

"""
當按鍵被觸發時的回呼函數。當按下有效鍵時停止錄音並儲存音訊。
:參數 key: pynput.keyboard.Key, 被按下的鍵
:回傳: None
"""
def on_press(key):
    global stop_flag
    try:
        key_char = getattr(key, 'char', None)
        if key_char is not None:  # 確保按下的是有效鍵
            save_sample()
            stop_flag = True  # 停止錄音
            return False  # 停止鍵盤監聽
    except AttributeError:
        pass

"""
啟動錄音和鍵盤監聽。這是主要的錄音與預測測試音訊的流程函式。
:參數 queue: queue.Queue, 用於傳遞訊息的佇列
:回傳: str, 音訊檔案的儲存路徑
"""
def get_test_audio(queue):
    global stop_flag, message_queue, frames
    message_queue = queue  # 將佇列傳遞給全域變數
    stop_flag = False
    frames = []  # 清空錄音緩衝區

    # 啟動鍵盤監聽器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # 啟動錄音執行緒
    audio_thread = threading.Thread(target=start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()

    listener.join()  # 等待鍵盤監聽完成
    stop_flag = True
    audio_thread.join(timeout=3.0)

    # 儲存錄音
    return save_sample()

"""
預測測試音訊的結果。該函式會錄製測試音訊、提取 MFCC 特徵，並使用 CNN 模型進行預測。
:參數 key_list: list, 用來對應預測結果的按鍵列表
:參數 queue: queue.Queue, 用於傳遞訊息的佇列
:回傳: str, 預測的按鍵
"""
def predict(key_list, queue):
    global message_queue
    message_queue = queue

    message_queue.put("開始錄製測試音訊...")
    audio_file = get_test_audio(message_queue)

    message_queue.put("提取測試音訊的特徵...")
    x_test = extract_mfcc_matrix(audio_file)
    if x_test is None:
        message_queue.put("無法提取測試音訊的特徵，請檢查音訊檔案。")
        return None

    message_queue.put("載入模型...")
    cnn_model = tf.keras.models.load_model('model.keras')

    message_queue.put("開始進行預測...")
    result = cnn_model.predict(x_test.reshape(1, x_test.shape[0], x_test.shape[1], 1))
    predicted_key = key_list[np.argmax(result)]
    message_queue.put(f"預測結果: {predicted_key}")
    return predicted_key
