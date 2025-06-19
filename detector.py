import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Toplevel, Label, Menu
import time
import os

class Detector:
    def __init__(self):
        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
        ]
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

    def _is_close(self, det1, det2, tolerance=30):
        x1, y1, w1, h1 = det1
        x2, y2, w2, h2 = det2
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        return np.hypot(cx1 - cx2, cy1 - cy2) < tolerance

    def _merge_detections(self, detections_list):
        merged = []
        for i in range(len(detections_list)):
            for rect1 in detections_list[i]:
                is_unique = True
                for rect2 in merged:
                    if self._is_close(rect1, rect2, tolerance=40):
                        is_unique = False
                        break
                if is_unique:
                    merged.append(rect1)
        return merged

    def _detect_and_display(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_faces = [cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50)) for cascade in self.face_cascades]
        merged_faces = self._merge_detections(all_faces)

        for (x, y, w, h) in merged_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes1 = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            eyes2 = self.left_eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            eyes3 = self.right_eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            all_eyes = [eyes1, eyes2, eyes3]
            merged_eyes = self._merge_detections(all_eyes)

            for (ex, ey, ew, eh) in merged_eyes:
                if ey + eh < h * 0.6:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return frame


    def video_detection(self, video_path, output_path=''):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nie udało się otworzyć pliku wideo: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_path == '':
            output_path = os.path.splitext(video_path)[0] + "_detected.mp4"
        elif os.path.isdir(output_path):
            output_path = os.path.join(output_path, base_name + "_detected.mp4")

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cv2.namedWindow("Analiza Wideo", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_OPENGL)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                raise ValueError("Bład przy przechwytywaniu obrazu z video.")
            
            frame = self._detect_and_display(frame)
            cv2.putText(frame, "ESC aby zakonczyc", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            out.write(frame)
            cv2.imshow("Video Detection", frame)

            if cv2.getWindowProperty("Video Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to: {output_path}")


    def live_detection(self):
        cap = cv2.VideoCapture(0)
        window_name = "Live Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_OPENGL)

        while True:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Bład przy przechwytywaniu obrazu z kamery")
            
            frame = self._detect_and_display(frame)
            cv2.putText(frame, "Naciśnij ESC aby zakonczyc", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()
        




if __name__ == "__main__":
    ...