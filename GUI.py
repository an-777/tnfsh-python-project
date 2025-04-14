import tkinter as tk
import threading
import queue
import os
from audio import get_audio
from cnn import train_model
from predict import predict

"""
收集輸入的字母並開始錄製樣本。
:參數 None
:回傳: None
"""
def collect_samples():
    key = input_box.get()
    input_box.config(state=tk.DISABLED)

    global audio_thread
    audio_thread = threading.Thread(target=get_audio, args=(key, output_queue))
    audio_thread.start()

"""
訓練模型。
:參數 None
:回傳: None
"""
def train_model_():
    key_list = list(input_box.get())
    if not key_list:
        output_queue.put("請輸入有效的字母列表以進行訓練。")
        return

    sample_dir = "samples"
    required_samples = len(key_list) * 10
    collected_samples = len([f for f in os.listdir(sample_dir) if f.endswith(".wav")]) if os.path.exists(sample_dir) else 0

    if collected_samples < required_samples:
        output_queue.put(f"樣本不足。需要 {required_samples} 個樣本，但僅有 {collected_samples} 個。")
        return

    def train_task():
        output_queue.put("開始訓練模型...")
        train_model(key_list)
        output_queue.put("模型訓練完成並儲存為 model.keras")

    train_thread = threading.Thread(target=train_task)
    train_thread.start()

"""
執行預測任務。
:參數 None
:回傳: None
"""
def predict_test_():
    key_list = list(input_box.get())
    if not key_list:
        return

    def predict_task():
        predict(key_list, output_queue)

    predict_thread = threading.Thread(target=predict_task)
    predict_thread.start()

"""
更新輸出區域顯示的訊息，從佇列中獲取並顯示最新的輸出。
:參數 None
:回傳: None
"""
def update_output():
    try:
        while True:
            message = output_queue.get_nowait()
            if "label" in message or "arg" in message or "getitem" in message:
                continue
            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, message + "\n")
            output_text.yview(tk.END)
            output_text.config(state=tk.DISABLED)
    except queue.Empty:
        pass
    root.after(100, update_output)

"""
將訊息顯示到 GUI 介面上，確保在主執行緒中執行更新。
:參數 message: str, 要顯示的訊息
:參數 *args: 任意數量的額外參數
:參數 **kwargs: 任意數量的額外關鍵字參數
:回傳: None
"""
def print_to_gui(message, *args, **kwargs):
    if root:
        root.after(0, lambda: output_text.config(state=tk.NORMAL) or output_text.insert(tk.END, message + "\n") or output_text.yview(tk.END) or output_text.config(state=tk.DISABLED))

"""
視窗關閉事件處理函式，停止錄製執行緒並清理所有執行緒。
:參數 None
:回傳: None
"""
def on_close():
    global audio_thread

    if audio_thread and audio_thread.is_alive():
        audio_thread.join(timeout=3.0)

    for thread in threading.enumerate():
        if thread != threading.main_thread():
            thread.join(timeout=3.0)

    output_queue.put("所有執行緒已停止，關閉視窗。")
    root.destroy()
    os._exit(0)

"""
GUI 設計
"""
root = tk.Tk()
root.title('鍵盤聲音預設按鍵')
root.geometry('300x400')

title = tk.Label(root, text='鍵盤聲音預設按鍵', font=(20))
title.pack()                                

title = tk.Label(root, text='輸入採集樣本字母(格式: abcd...)')
title.pack()                                

input_box = tk.Entry(root)
input_box.pack()

get_ = tk.Button(root, text='採集樣本', command=collect_samples)
get_.pack()
train = tk.Button(root, text='訓練模型', command=train_model_)
train.pack()

predict_ = tk.Button(root, text='預測', command=predict_test_)
predict_.pack()

output_text = tk.Text(root, height=10, width=40)
output_text.pack()
output_text.config(state=tk.DISABLED)

output_queue = queue.Queue()

audio_thread = None

update_output()

root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
