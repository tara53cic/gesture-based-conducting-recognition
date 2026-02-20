#Skripta za kreiranje dataset-a

#Snimamo web kamerom dok dirigujemo specifičan tempo i vrstu takta,
#detektujemo ruku pomoću MediaPipe-a i čuvamo koordinate landmarkova
#zajedno sa labelom i BPM-om u CSV fajl.

#Treba ponoviti za različite kombinacije taktova i BPM-ova da bismo imali raznovrstan dataset.

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import time

#Parametri

LABEL = "4_4"
BPM_TARGET = 180
RECORDING_DURATION = 90  
MODEL_PATH = 'hand_landmarker.task' 
output_file = f"data_{LABEL}_{BPM_TARGET}.csv"

# Inicijalizacija MediaPipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def display_text(image, text, color=(0, 255, 0), scale=1.5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (50, image.shape[0] // 2), font, scale, color, 3)

#Odbrojavanje pre početka snimanja
for i in range(3, 0, -1):
    start_wait = time.time()
    while time.time() - start_wait < 1:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_text(frame, f"SPREMITE SE: {i}", (0, 255, 255))
        cv2.imshow('Recording', frame)
        cv2.waitKey(1)

#Snimanje
data_buffer = []
start_time = time.time()
finished_naturally = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    #Da li smo prekinuli snimanje ili je vreme isteklo
    elapsed = time.time() - start_time
    if elapsed >= RECORDING_DURATION:
        finished_naturally = True
        break

    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Detekcija
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        # Uzimamo prvu detektovanu ruku
        hand_lms = detection_result.hand_landmarks[0]
        
        #Čuvamo vreme, labelu za vrstu takta i bpm, i koordinate landmarkova
        row = [time.time(), LABEL, BPM_TARGET]
        for lm in hand_lms:
            # Crtanje krugova na landmarkovima da bi znali da li MediaPipe detektuje ruku kako treba
            x_px = int(lm.x * frame.shape[1])
            y_px = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x_px, y_px), 3, (0, 255, 0), -1)
            row += [lm.x, lm.y, lm.z]
        data_buffer.append(row)


    cv2.putText(frame, f"Vreme: {int(RECORDING_DURATION - elapsed)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Recording', frame)
    
    #ako prekinemo snimanje pre vremena, nećemo sačuvati podatke
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Prekinuto. Podaci NISU sačuvani.")
        break

if finished_naturally:

    end_wait = time.time()
    while time.time() - end_wait < 2:
        ret, frame = cap.read()
        display_text(frame, "ZAVRSENO! CUVANJE...", (255, 255, 0))
        cv2.imshow('Recording', frame)
        cv2.waitKey(1)

    # Čuvamo podatke u CSV
    header = ['timestamp', 'label', 'bpm'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_buffer)
    print(f"Podaci uspešno sačuvani u {output_file}")

cap.release()
cv2.destroyAllWindows()