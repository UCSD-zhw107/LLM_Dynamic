import time
import cv2
from pynput import keyboard



# 全局变量，标记是否按下 ESC
running = True

def on_press(key):
    global running
    if key == keyboard.Key.esc:
        print("ESC pressed, exiting loop...")
        running = False  # 让 `while` 循环退出

# 启动键盘监听
listener = keyboard.Listener(on_press=on_press)
listener.start()

# 你的主循环
print("Press ESC to exit...")
while running:
    print("Looping...")
    time.sleep(0.5)  # 模拟你的主逻辑

print("Cleaning up...")
listener.stop()  # 关闭监听