import cv2
import face_recognition
import numpy as np
import pickle
import tkinter as tk
from tkinter import simpledialog

def register_face():
    name = simpledialog.askstring("Input", "Enter name:")
    user_id = simpledialog.askstring("Input", "Enter user ID:")
    if not name or not user_id:
        print("Name and ID are required")
        return

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Registration")

    face_encodings = []
    while len(face_encodings) < 5:
        ret, frame = cam.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            face_encodings.append(encoding)
            cv2.rectangle(frame, (face_locations[0][3], face_locations[0][0]), (face_locations[0][1], face_locations[0][2]), (0, 255, 0), 2)
            cv2.putText(frame, "Face captured", (face_locations[0][3], face_locations[0][0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Registration", frame)
        cv2.waitKey(1000)

    cam.release()
    cv2.destroyAllWindows()

    # Save face encodings
    if face_encodings:
        with open("face_encodings.pickle", "ab") as f:
            pickle.dump({"name": name, "id": user_id, "encodings": face_encodings}, f)

        print("Face registered successfully")

root = tk.Tk()
root.withdraw()
register_face()
