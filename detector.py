import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Toplevel, Label, Menu, ttk
import time
import os
import logging

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.face_cascades = [
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
        ]
        # self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

        self.scale_factor = 1.3
        self.min_neighbors = 5

    # to check if two detected regions overlap/are close
    # tolerance - aximum allowed pixel distance (default 60px)
    def _is_close(self, det1, det2, tolerance=60):
        x1, y1, w1, h1 = det1
        x2, y2, w2, h2 = det2
        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        return np.hypot(cx1 - cx2, cy1 - cy2) < tolerance

    # combines overlapping detections from different classifiers
    # avoids duplicate detections from multiple cascades
    def _merge_detections(self, detections_list):
        merged = []
        for i in range(len(detections_list)):
            for rect1 in detections_list[i]:
                is_unique = True
                for rect2 in merged:
                    if self._is_close(rect1, rect2, tolerance=80):
                        is_unique = False
                        break
                if is_unique:
                    merged.append(rect1)
        return merged


    def _detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # minNeighbors - detection confidence
        # minSize - minimum size of detecteble face
        all_faces = [
        cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        for cascade in self.face_cascades
    ]


        # merge detections from all cascades
        merged_faces = self._merge_detections(all_faces)

        for (x, y, w, h) in merged_faces:
            # mark found faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 5)
            # region of interest for eyes detection
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            l_eye = self.left_eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            r_eye = self.right_eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
            all_eyes = [l_eye, r_eye]
            merged_eyes = self._merge_detections(all_eyes)

            for (ex, ey, ew, eh) in merged_eyes:
                # consider eyes only in upper 0.4 part of the face
                # this will prevent marking mouth region detections
                if ey + eh < h * 0.6:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return frame


    def video_detection(self, progress: ttk.Progressbar, 
        percent_label: Label, top_level: Toplevel, 
        video_path, output_path=''):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nie udało się otworzyć pliku wideo: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Wideo: {video_path}, {frame_count} klatek, {fps:.2f} FPS, {width}x{height}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_path == '':
            output_path = os.path.splitext(video_path)[0] + "_detected.mp4"
        elif os.path.isdir(output_path):
            output_path = os.path.join(output_path, base_name + "_detected.mp4")

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self._detect(frame)
            out.write(processed)
            
            progress['value'] = (frame_idx / frame_count) * 100
            percent_label.config(text=f"{int((frame_idx / frame_count) * 100)}%")
            top_level.update()

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Przetworzono {frame_idx}/{frame_count} klatek")

        cap.release()
        out.release()
        logger.info(f"Zapisano: {output_path}")



    def live_detection(self):
        cap = cv2.VideoCapture(1)
        window_name = "Detekcja live"

        while True:
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Bład przy przechwytywaniu obrazu z kamery")
            
            frame = self._detect(frame)
            cv2.putText(frame, "Nacisnij ESC aby zakonczyc", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)

            cv2.imshow(window_name, frame)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()
        




if __name__ == "__main__":
    ...