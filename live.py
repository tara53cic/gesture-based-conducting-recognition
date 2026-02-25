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

#Biramo pesmu iz skupa koja je istog takta a najbližeg tempa 
def get_closest_song(target_takt, target_bpm):
    if not os.path.exists(SONGS_FOLDER):
        return None, 0
        
    files = [f for f in os.listdir(SONGS_FOLDER) if f.startswith(f"song_{target_takt}")]
    if not files:
        return None, 0

    available_bpms = []
    for f in files:
        match = re.search(r'_(\d+)\.mp3', f)
        if match:
            available_bpms.append(int(match.group(1)))

    if not available_bpms:
        return None, 0

    closest_bpm = min(available_bpms, key=lambda x: abs(x - target_bpm))
    path = os.path.join(SONGS_FOLDER, f"song_{target_takt}_{closest_bpm}.mp3")
    
    return path, closest_bpm

def play_song(path):
    try:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        pygame.mixer.init() 
        pygame.mixer.music.load(path)
        pygame.mixer.music.play(-1)
        return True
    except:
        return False

# Procesiranje podataka isto kao u treningu modela
def process_live_buffer(buffer):

    raw_times = np.array([f[0] for f in buffer])
    raw_data = np.array([f[1] for f in buffer])

    #Ako nismo sakupili 4 sekunde (jedan prozor) onda još ne procesiramo
    if raw_times[-1] - raw_times[0] < 4.0:
        return None

    #Uzimamo poslednje 4 sekunde podataka
    mask = raw_times >= (raw_times[-1] - 4.0)
    filtered_times = raw_times[mask]
    filtered_data = raw_data[mask]

    #Relativne koordinate u odnosu na zglob
    for i in range(21):
        filtered_data[:, i*3:i*3+3] -= filtered_data[:, 0:3]

    rel_time = filtered_times - filtered_times[0]
    new_time_steps = np.arange(0, 4.0, 1/TARGET_FPS)

    #Interpolacija na 120 frejmova
    resampled_data = []
    for col in range(63):
        f = interp1d(rel_time, filtered_data[:, col], kind='linear', fill_value="extrapolate")
        resampled_data.append(f(new_time_steps))

    resampled = np.array(resampled_data).T

    #Računamo brzinu- razliku između frejmova 
    velocity = np.diff(resampled, axis=0)
    velocity = np.vstack([velocity[0], velocity])
    resampled = np.concatenate([resampled, velocity], axis=1)

    return resampled.reshape(1, WINDOW_SIZE, 126)

# Učitavanje modela, MediaPipe detektora itd...
model = load_model(AI_MODEL_PATH, compile=False)
base_options = python.BaseOptions(model_asset_path=MP_TASK_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

frame_buffer = deque(maxlen=WINDOW_SIZE)

#Za praćenje istorije predikcija radi stabilizacije i računanja medijane tempa
task_history = deque(maxlen=7)
bpm_history = deque(maxlen=7)

#Stanja potrebna za UI
is_playing = False
show_stats = False
active_takt, active_bpm = None, 0
curr_t_name, curr_b_val = "N/A", 0

calibration_start = None
last_pred_time = 0
hand_unseen_counter = 0

mistake_takt, mistake_bpm = 0, 0
total_checks_takt, total_checks_bpm = 0, 0

flash_message = ""
flash_timer = 0

cap = cv2.VideoCapture(0)
pygame.mixer.init()

try:
    while cap.isOpened():
        #Čitamo frejm sa kamere
        ret, frame = cap.read()
        if not ret: break

        #Usklađivanje zbog MediaPipe logike
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)

        #Ako vidimo ruku crtamo koordinate i dodajemo u buffer, počinjemo kalibraciju
        if res.hand_landmarks:
            show_stats = False
            hand_unseen_counter = 0
            coords = []
            for lm in res.hand_landmarks[0]:
                coords.extend([lm.x, lm.y, lm.z])
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0, 255, 0), -1)
            frame_buffer.append((current_time, coords))
            if calibration_start is None: calibration_start = current_time
        
        #Brojimo koliko dugo nismo videli ruku jer gasimo muziku ako prestane taktiranje
        else:
            hand_unseen_counter += 1
            if hand_unseen_counter > HAND_LOSS_THRESHOLD:
                if is_playing:
                    pygame.mixer.music.stop()
                    is_playing = False
                    show_stats = True
                calibration_start = None
                frame_buffer.clear()

        # Predikcije takta i bpm ako imamo popunjen prozor od 4s
        if len(frame_buffer) >= WINDOW_SIZE and (current_time - last_pred_time) >= PREDICTION_INTERVAL:
            input_data = process_live_buffer(list(frame_buffer))
            if input_data is not None:
                p_task_raw, p_bpm_raw = model.predict(input_data, verbose=0)
                
                task_history.append(np.argmax(p_task_raw))
                bpm_history.append(p_bpm_raw[0][0])

                #trenutni takt je onaj koji se najčešće pojavio u istoriji predikcija,
                #a tempo je medijana predikcija tempa, radi bolje stabilizacije rezultata

                curr_t_idx = max(set(task_history), key=task_history.count)
                curr_t_name = LABEL_NAMES[curr_t_idx]
                curr_b_val = np.median(bpm_history)
                last_pred_time = current_time

                # Ako smo završili kalibraciju, biramo pesmu i startujemo muziku
                if not is_playing and calibration_start and (current_time - calibration_start) >= CALIBRATION_DURATION:
                    path, song_fixed_bpm = get_closest_song(curr_t_name, curr_b_val)
                    if path and play_song(path):
                        active_takt = curr_t_name
                        active_bpm = song_fixed_bpm
                        is_playing = True
                        mistake_takt, mistake_bpm = 0, 0
                        total_checks_takt, total_checks_bpm = 0, 0
                
                #Proveravamo da li smo u dozvoljenom rasponu tempa i takta,
                #brojimo i greške i ukupan broj provera da bismo znali % ba kraju

                elif is_playing:
                    total_checks_takt += 1
                    if curr_t_name != active_takt:
                        mistake_takt += 1

                    total_checks_bpm += 1
                    diff = curr_b_val - active_bpm 

                    #Da li taktiramo suviše brzo ili sporo?
                    if abs(diff) > BPM_ERROR_THRESHOLD:
                        mistake_bpm += 1
                        flash_message = "Too Fast" if diff > 0 else "Too Slow"
                        flash_timer = 15

        # UI
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
            cv2.putText(frame, "CALIBRATING...", (w//2-100, h//2), 1, 1.5, (0, 255, 255), 2)

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