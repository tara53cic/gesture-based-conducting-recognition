import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d
import time
import os
import pygame
import re
from collections import deque
import threading

# Parametri i konfiguracija
AI_MODEL_PATH = 'conductor_ai.h5'
MP_TASK_PATH = 'hand_landmarker.task'
SONGS_FOLDER = 'songs'

TARGET_FPS = 30
WINDOW_SECONDS = 4
WINDOW_SIZE = TARGET_FPS * WINDOW_SECONDS
LABEL_NAMES = ["2_4", "3_4", "4_4"]

CALIBRATION_DURATION = 5.0
PREDICTION_INTERVAL = 0.5
HAND_LOSS_THRESHOLD = 15
BPM_ERROR_THRESHOLD = 12

# Globalne promenljive za komunikaciju sa thread-om
thread_lock = threading.Lock()
is_predicting = False
latest_prediction = {"t_name": "N/A", "b_val": 0, "new_data": False}

def get_closest_song(target_takt, target_bpm):
    if not os.path.exists(SONGS_FOLDER): return None, 0
    files = [f for f in os.listdir(SONGS_FOLDER) if f.startswith(f"song_{target_takt}")]
    if not files: return None, 0
    available_bpms = []
    for f in files:
        match = re.search(r'_(\d+)\.mp3', f)
        if match: available_bpms.append(int(match.group(1)))
    if not available_bpms: return None, 0
    closest_bpm = min(available_bpms, key=lambda x: abs(x - target_bpm))
    path = os.path.join(SONGS_FOLDER, f"song_{target_takt}_{closest_bpm}.mp3")
    return path, closest_bpm

def play_song(path):
    try:
        if pygame.mixer.get_init(): pygame.mixer.quit()
        pygame.mixer.init() 
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1)
        return True
    except: return False

def process_live_buffer(buffer):
    # Preuzimanje 4s
    raw_times = np.array([f[0] for f in buffer])
    raw_data = np.array([f[1] for f in buffer])
    if raw_times[-1] - raw_times[0] < 4.0: return None
    mask = raw_times >= (raw_times[-1] - 4.0)
    filtered_times = raw_times[mask]
    filtered_data = raw_data[mask]
    
    # Normalizacija koordinata
    wrist = filtered_data[:, 0:3]
    for i in range(21):
        filtered_data[:, i*3:i*3+3] -= wrist

    rel_time = filtered_times - filtered_times[0]
    new_time_steps = np.linspace(0, 4.0, WINDOW_SIZE)

    # Interpolacija i brzina
    f_interp = interp1d(rel_time, filtered_data, axis=0, kind='linear', fill_value="extrapolate")
    resampled = f_interp(new_time_steps)
    velocity = np.diff(resampled, axis=0)
    velocity = np.vstack([velocity[0], velocity])
    resampled = np.concatenate([resampled, velocity], axis=1)
    return resampled.reshape(1, WINDOW_SIZE, 126)

# Worker funkcija za AI predikciju u pozadini

def ai_prediction_worker(buffer_copy, model_ref):
    global is_predicting, latest_prediction
    try:
        input_data = process_live_buffer(buffer_copy)
        if input_data is not None:
            # Model poziv i predikcija
            p_task_raw, p_bpm_raw = model_ref(input_data, training=False)
            t_idx = np.argmax(p_task_raw.numpy())
            b_val = p_bpm_raw.numpy()[0][0]
            
            with thread_lock:
                latest_prediction["t_name"] = LABEL_NAMES[t_idx]
                latest_prediction["b_val"] = b_val
                latest_prediction["new_data"] = True
    finally:
        is_predicting = False

# Inicijalizacija modela
model = load_model(AI_MODEL_PATH, compile=False)
base_options = python.BaseOptions(model_asset_path=MP_TASK_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

frame_buffer = deque(maxlen=400)
task_history = deque(maxlen=10)
bpm_history = deque(maxlen=10)

is_playing, show_stats = False, False
active_takt, active_bpm = None, 0
curr_t_name, curr_b_val = "N/A", 0
calibration_start, last_pred_time = None, 0
hand_unseen_counter = 0
mistake_takt, mistake_bpm, total_checks_takt, total_checks_bpm = 0, 0, 0, 0
flash_message, flash_timer = "", 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
pygame.mixer.init()

try:
    while cap.isOpened():
        #Bacamo stare frejmove iz bafera operativnog sistema
        for _ in range(2): cap.grab()
        ret, frame = cap.retrieve()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)

        if res.hand_landmarks:
            show_stats = False
            hand_unseen_counter = 0
            coords = []
            for lm in res.hand_landmarks[0]:
                coords.extend([lm.x, lm.y, lm.z])
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0, 255, 0), -1)
            
            frame_buffer.append((current_time, coords))
            while frame_buffer and (current_time - frame_buffer[0][0]) > 4.5:
                frame_buffer.popleft()

            has_4s_data = (frame_buffer[-1][0] - frame_buffer[0][0]) >= 4.0

            if has_4s_data:
                if calibration_start is None and not is_playing:
                    calibration_start = current_time

                # Pokretanje predikcije u drugom thread-u ako je vreme
                if (current_time - last_pred_time) >= PREDICTION_INTERVAL and not is_predicting:
                    is_predicting = True
                    last_pred_time = current_time
                    # Saljemo kopiju bafera thread-u
                    buffer_copy = list(frame_buffer)
                    threading.Thread(target=ai_prediction_worker, args=(buffer_copy, model), daemon=True).start()

                # Preuzimanje rezultata iz pozadinskog thread-a
                with thread_lock:
                    if latest_prediction["new_data"]:
                        task_history.append(latest_prediction["t_name"])
                        bpm_history.append(latest_prediction["b_val"])
                        
                        curr_t_name = max(set(task_history), key=list(task_history).count)
                        curr_b_val = np.median(bpm_history)
                        latest_prediction["new_data"] = False

                        # Logika za greske
                        if is_playing:
                            total_checks_takt += 1
                            if curr_t_name != active_takt: mistake_takt += 1
                            total_checks_bpm += 1
                            diff = curr_b_val - active_bpm 
                            if abs(diff) > BPM_ERROR_THRESHOLD:
                                mistake_bpm += 1
                                flash_message = "Too Fast" if diff > 0 else "Too Slow"
                                flash_timer = 15

                # Kalibracija
                if not is_playing and calibration_start and (current_time - calibration_start) >= CALIBRATION_DURATION:
                    if len(bpm_history) > 0:
                        path, song_fixed_bpm = get_closest_song(curr_t_name, curr_b_val)
                        if path and play_song(path):
                            active_takt, active_bpm = curr_t_name, song_fixed_bpm
                            is_playing = True
                            mistake_takt, mistake_bpm, total_checks_takt, total_checks_bpm = 0, 0, 0, 0
                    else: calibration_start = None

        else:
            hand_unseen_counter += 1
            if hand_unseen_counter > HAND_LOSS_THRESHOLD:
                if is_playing:
                    pygame.mixer.music.stop()
                    is_playing, show_stats = False, True
                calibration_start = None
                frame_buffer.clear()

        #UI
        if is_playing:
            cv2.rectangle(frame, (10, 5), (w//2 - 10, 55), (0, 0, 0), -1)
            cv2.putText(frame, f"YOU: {curr_t_name} @ {int(curr_b_val)} BPM", (20, 40), 1, 1.2, (0, 255, 0), 2)
            cv2.rectangle(frame, (w//2 + 10, 5), (w - 10, 55), (0, 0, 0), -1)
            song_txt = f"SONG: {active_takt} @ {int(active_bpm)} BPM"
            t_size = cv2.getTextSize(song_txt, 1, 1.2, 2)[0]
            cv2.putText(frame, song_txt, (w - 20 - t_size[0], 40), 1, 1.2, (255, 255, 255), 2)
            bw, bh = 260, 100
            cv2.rectangle(frame, (w-bw-10, h-bh-10), (w-10, h-10), (0,0,0), -1)
            cv2.putText(frame, "Mistakes:", (w-bw+10, h-bh+25), 1, 1.1, (255,255,255), 2)
            cv2.putText(frame, f"Meter: {mistake_takt}", (w-bw+10, h-bh+55), 1, 0.9, (200,200,200), 1)
            cv2.putText(frame, f"Tempo: {mistake_bpm}", (w-bw+10, h-bh+85), 1, 0.9, (200,200,200), 1)
        elif calibration_start:
            time_left = max(0, 5.0 - (current_time - calibration_start))
            cv2.putText(frame, f"CALIBRATING... {time_left:.1f}s", (w//2-150, h//2), 1, 1.5, (0, 255, 255), 2)
        elif not is_playing and res.hand_landmarks:
            cv2.putText(frame, "WINDING UP...", (w//2-100, h//2), 1, 1.5, (255, 100, 0), 2)

        if flash_timer > 0:
            cv2.putText(frame, flash_message, (w//2-80, h-100), 1, 2, (0,0,255), 3)
            flash_timer -= 1

        if show_stats:
            cv2.rectangle(frame, (0,0), (w,h), (0,0,0), -1)
            t_acc = 100 * (1 - mistake_takt/max(1, total_checks_takt))
            b_acc = 100 * (1 - mistake_bpm/max(1, total_checks_bpm))
            cv2.putText(frame, "SESSION STATS", (w//2-140, 150), 1, 2, (0,255,0), 3)
            cv2.putText(frame, f"Meter Accuracy: {t_acc:.1f}%", (w//2-180, 230), 1, 1.2, (255,255,255), 2)
            cv2.putText(frame, f"Tempo Accuracy: {b_acc:.1f}%", (w//2-180, 280), 1, 1.2, (255,255,255), 2)
            cv2.putText(frame, "Show hands to start again", (w//2-150, 400), 1, 1, (150,150,150), 1)

        cv2.imshow('AI Conductor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    detector.close()
    cap.release()
    cv2.destroyAllWindows()