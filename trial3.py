# python -m venv venv
# venv\Scripts\activate

import sys
import cv2
import math
import time
import numpy as np
import winsound
import mediapipe as mp
from PySide6.QtCore import QThread, Signal, Slot, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame, QSizePolicy)
import pyqtgraph as pg  


# bg worker thread
class VideoThread(QThread):

    change_pixmap_signal = Signal(QImage, str, bool, float, float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # establish baselines for calibration
        self.calibrated_eye_dist = 90.0
        self.calibrated_shoulder_y = 0.5
        self.is_calibrated = False

        # threshold limits 
        self.DISTANCE_THRESHOLD_OFFSET = 25  # Pixels closer than baseline allowed
        self.SLOUCH_THRESHOLD_OFFSET = 0.06  # Percentage drop allowed

        self.bad_posture_start_time = None
        self.ALERT_DELAY_SECONDS = 2.0 

    @Slot(float, float)
    def calibrate(self, current_eye_dist, current_shoulder_y):
        if current_eye_dist > 0:
            self.calibrated_eye_dist = current_eye_dist
            self.calibrated_shoulder_y = current_shoulder_y
            self.is_calibrated = True

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        mp_drawing = mp.solutions.drawing_utils

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: continue

            frame = cv2.flip(frame, 1)
            h, w, ch = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            status_text = "Calibrate System First" if not self.is_calibrated else "Good Posture"
            is_alert = False
            eye_distance, shoulder_y = 0.0, 0.0

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                display_frame = frame.copy()
                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=1,
                        circle_radius=1
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1
                    )
                )
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # metrics gathering
                le = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
                re = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
                eye_distance = math.hypot((le.x - re.x) * w, (le.y - re.y) * h)
                
                ls_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                rs_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

                ls_x= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                rs_x= landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x

                shoulder_y = (ls_y + rs_y) / 2.0

                slump_angle= math.degrees(math.atan2((ls_y -rs_y),(ls_x - rs_x)))
                slump_thresh= 10 #degrees
                

                if self.is_calibrated:
                    # deviation calculations
                    is_too_close = eye_distance > (self.calibrated_eye_dist + self.DISTANCE_THRESHOLD_OFFSET)
                    is_slouching = shoulder_y > (self.calibrated_shoulder_y + self.SLOUCH_THRESHOLD_OFFSET)
                    is_slumping= abs(slump_angle) > slump_thresh

                    if is_too_close:
                        status_text = "TOO CLOSE TO SCREEN!"
                        is_alert = True
                    elif is_slouching:
                        status_text = "SLOUCHING DETECTED!"
                        is_alert = True
                    elif is_slumping:
                        status_text = "SLUMPING DETECTED!"
                        is_alert = True

                    # alert timer check
                    if is_alert:
                        if self.bad_posture_start_time is None:
                            self.bad_posture_start_time = time.time()
                        elif time.time() - self.bad_posture_start_time > self.ALERT_DELAY_SECONDS:
                            winsound.Beep(1200, 80)
                    else:
                        self.bad_posture_start_time = None

            qt_image = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image, status_text, is_alert, eye_distance, shoulder_y)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProPosture Engine v2.0")
        self.resize(1100, 650)
        
        # current capture metrics for calibration 
        self.latest_eye_dist = 0.0
        self.latest_shoulder_y = 0.0
        self._latest_pixmap = None

        #styling
        self.setStyleSheet("""
            QMainWindow { background-color: #0B0E14; }
            QFrame#MainCard { 
                background-color: #161B22; 
                border-radius: 16px; 
                border: 1px solid #30363D;
            }
            QLabel { color: #C9D1D9; font-family: 'Segoe UI', system-ui; }
            QLabel#Title { font-size: 24px; font-weight: 800; color: #58A6FF; }
            QPushButton {
                background-color: #21262D;
                color: #58A6FF;
                border: 1px solid #30363D;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #30363D; color: #79C0FF; }
            QPushButton:pressed { background-color: #1F6FEB; color: white; }
        """)

        # main layout separation 
        main_layout = QHBoxLayout()
        container_widget = QWidget()
        container_widget.setLayout(main_layout)
        self.setCentralWidget(container_widget)

        # left sidebar
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, stretch=3)

        title_lbl = QLabel("PROPOSTURE AI")
        title_lbl.setObjectName("Title")
        left_panel.addWidget(title_lbl)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setStyleSheet("border-radius: 12px; border: 2px solid #30363D; background-color: #000000;")
        left_panel.addWidget(self.image_label)

        self.calibrate_btn = QPushButton("🎯 Calibrate Perfect Sitting Posture")
        self.calibrate_btn.clicked.connect(self.trigger_calibration)
        left_panel.addWidget(self.calibrate_btn)

        #right panel
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, stretch=2)

        # notif card
        self.status_card = QFrame()
        self.status_card.setObjectName("MainCard")
        sc_layout = QVBoxLayout(self.status_card)
        self.status_text_label = QLabel("Sit straight and click calibrate...")
        self.status_text_label.setAlignment(Qt.AlignCenter)
        self.status_text_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #8B949E;")
        sc_layout.addWidget(self.status_text_label)
        right_panel.addWidget(self.status_card, stretch=1)

        # pyqtgraph plot widget
        self.graph_card = QFrame()
        self.graph_card.setObjectName("MainCard")
        gc_layout = QVBoxLayout(self.graph_card)
        
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('#161B22')
        self.graph_widget.setTitle("Posture Stability Score Timeline", color="#C9D1D9", size="12pt")
        self.graph_widget.showGrid(x=False, y=True, alpha=0.2)
        self.graph_widget.setYRange(-10, 110)
        
        
        self.history_data = list(np.zeros(100))
        self.data_line = self.graph_widget.plot(self.history_data, pen=pg.mkPen(color='#58A6FF', width=3))
        
        gc_layout.addWidget(self.graph_widget)
        right_panel.addWidget(self.graph_card, stretch=3)

       
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_dashboard)
        self.thread.start()


    def trigger_calibration(self):
        self.thread.calibrate(self.latest_eye_dist, self.latest_shoulder_y)
        self.calibrate_btn.setText("✓ Recalibrate Settings Baseline")


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._latest_pixmap is not None and not self.image_label.size().isEmpty():
            scaled_pixmap = self._latest_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    @Slot(QImage, str, bool, float, float)
    def update_dashboard(self, qt_image, status_text, is_alert, eye_dist, shoulder_y):
       
        self.latest_eye_dist = eye_dist
        self.latest_shoulder_y = shoulder_y

        self._latest_pixmap = QPixmap.fromImage(qt_image)
        if not self.image_label.size().isEmpty():
            scaled_pixmap = self._latest_pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setPixmap(self._latest_pixmap)
        self.status_text_label.setText(status_text)

        score = 0 if is_alert else (100 if self.thread.is_calibrated else 50)
        
        self.history_data.pop(0)
        self.history_data.append(score)
        self.data_line.setData(self.history_data)


        if not self.thread.is_calibrated:
            self.status_card.setStyleSheet("QFrame#MainCard { background-color: #21262D; border: 1px solid #30363D; }")
            self.status_text_label.setStyleSheet("color: #8B949E; font-size: 20px; font-weight: bold;")
            self.data_line.setPen(pg.mkPen(color='#8B949E', width=2))
        elif is_alert:
            self.status_card.setStyleSheet("QFrame#MainCard { background-color: #3c1e22; border: 2px solid #f85149; }")
            self.status_text_label.setStyleSheet("color: #f85149; font-size: 22px; font-weight: bold;")
            self.data_line.setPen(pg.mkPen(color='#f85149', width=3))
        else:
            self.status_card.setStyleSheet("QFrame#MainCard { background-color: #161B22; border: 1px solid #30363D; }")
            self.status_text_label.setStyleSheet("color: #58A6FF; font-size: 20px; font-weight: bold;")
            self.data_line.setPen(pg.mkPen(color='#58A6FF', width=3))

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())