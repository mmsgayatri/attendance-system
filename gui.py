import tkinter as tk
from tkinter import simpledialog
import subprocess

def register_face():
    name = simpledialog.askstring("Input", "Enter name:")
    user_id = simpledialog.askstring("Input", "Enter user ID:")
    if name and user_id:
        subprocess.run(["python", "register_face.py"], input=f"{name}\n{user_id}\n", text=True)

def take_attendance():
    subprocess.run(["python", "recognize_faces.py"])

root = tk.Tk()
root.title("Face Recognition Attendance System")

register_button = tk.Button(root, text="Register Face", command=register_face)
register_button.pack(pady=20)

attendance_button = tk.Button(root, text="Take Attendance", command=take_attendance)
attendance_button.pack(pady=20)

root.mainloop()
