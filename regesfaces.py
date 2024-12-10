import face_recognition
import cv2
import numpy as np
import pickle
import pandas as pd
import datetime
import os

def mark_attendance(name):
    now = datetime.datetime.now()
    today_date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    if os.path.exists("attendance.csv") and os.path.getsize("attendance.csv") > 0:
        try:
            df = pd.read_csv("attendance.csv")
            if any((df['Name'] == name) & (df['Date'] == today_date)):
                return False
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    new_entry = pd.DataFrame([{"Name": name, "Date": today_date, "Time": current_time}])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv("attendance.csv", index=False)

    return True

def recognize_faces():
    with open("face_encodings.pickle", "rb") as f:
        data = []
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    known_encodings = [item['encodings'][0] for item in data]
    known_names = [item['name'] for item in data]

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Recognition")

    frame_skip = 3
    frame_count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                message = ""

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                    if mark_attendance(name):
                        message = f"Attendance marked for {name}"
                    else:
                        message = f"Attendance already taken for {name}"

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                color = (0, 255, 0) if message.startswith("Attendance marked") else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, message, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        frame_count += 1
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

recognize_faces()
