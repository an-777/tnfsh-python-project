import pyaudio
import wave
import os
import time
from pynput import keyboard
import threading
import queue

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
key_list = []
press_counts = {}
current_key_index = 0
stop_flag = False
message_queue = None

"""
開始錄音，並持續錄取音訊直到停止標誌被設置為 True。
:參數 None
:回傳: None
"""
def start_audio_stream():
    global p, stream, stop_flag, key_list
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        message_queue.put("錄音開始...")
        message_queue.put(f"按下 {key_list[0]} 10 下(間隔0.2秒以上)")

        while not stop_flag:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
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
儲存當前錄音片段為 WAV 檔案。
:參數 key_char: str, 按下的鍵字符號
:參數 count: int, 該鍵的錄音次數
:回傳: str, 儲存的檔案名稱
"""
def save_sample(key_char, count):
    global frames, CHUNK, RATE
    output_dir = "samples"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    time.sleep(0.1)
 
    filename = os.path.join(output_dir, f"{key_char}_{count}.wav")
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames[-int(RATE / CHUNK * 0.2): ]))
    wf.close()
    
    message_queue.put(f"{key_char}採樣成功({count}/10)")
    return filename

"""
當按鍵被觸發時的回呼函數，會處理每個按鍵的偵測與錄音。
:參數 key: pynput.keyboard.Key, 被按下的鍵
:回傳: None
"""
def on_press(key):
    global stream, p, current_key_index, press_counts, stop_flag
    try:      
        key_char = getattr(key, 'char', None)
        if key_char is None:  # 當按下的不是字母鍵時
            return
        
        current_key = key_list[current_key_index]
        
        if key_char == current_key:
            press_counts[key_char] += 1
            save_sample(key_char, press_counts[key_char])
            
            if press_counts[key_char] == 10 and current_key_index == len(key_list) - 1:
                message_queue.put("錄音完成")
                stop_flag = True
                return False
            elif press_counts[key_char] >= 10 and current_key_index < len(key_list) - 1:
                current_key_index += 1
                message_queue.put(f"按下 {key_list[current_key_index]} 10 下(間隔0.2秒以上)")

    except AttributeError:
        pass

"""
啟動錄音和鍵盤監聽，並保存每個鍵的音訊樣本。
:參數 key: str, 需要錄音的鍵字符號
:參數 queue: queue.Queue, 用來傳遞訊息的佇列
:回傳: None
"""
def get_audio(key, queue):
    global stop_flag, key_list, press_counts, current_key_index, message_queue
    message_queue = queue  # 將佇列傳遞給全域變數
    key_list = list(key)  # 確保 key 是列表類型
    press_counts = {key: 0 for key in key_list}
    current_key_index = 0

    # 啟動鍵盤監聽器
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # 啟動錄音執行緒
    audio_thread = threading.Thread(target=start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start()
    
    listener.join()
    stop_flag = True
    audio_thread.join(timeout=3.0)
    
    listener.stop()
