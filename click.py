from pynput import mouse,keyboard
import threading
import time

keyboard_controller = keyboard.Controller()
def simulate_ctrl_end():
    keyboard_controller.press(keyboard.Key.ctrl)
    keyboard_controller.press(keyboard.Key.end)
    keyboard_controller.release(keyboard.Key.end)
    keyboard_controller.release(keyboard.Key.ctrl)
    print("Ctrl + End")

def on_click(x, y, button, pressed):
    if button == mouse.Button.right and pressed:
        simulate_ctrl_end()

def on_press(key):
    if key == keyboard.Key.space:
        simulate_ctrl_end()

with keyboard.Listener(on_press=on_press) as listener:
    print("监听空格，模拟Ctrl+End")
    listener.join()
