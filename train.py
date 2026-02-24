import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.metrics import confusion_matrix
import random

#Postavljanje seed-a za reproducibilnost
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.config.experimental.enable_op_determinism()

#Parametri
DATA_FOLDER = 'dataset'
TEST_FOLDER = 'test_data' 
TARGET_FPS = 30
WINDOW_SECONDS = 4
WINDOW_SIZE = TARGET_FPS * WINDOW_SECONDS
STEP_SIZE = 10
LABEL_MAP = {"2_4": 0, "3_4": 1, "4_4": 2}

def preprocess_file(file_path):
    df = pd.read_csv(file_path)
    
    #Relativne koordinate, normalizujemo u odnosu na zglob
    for i in range(21):
        df[f'x{i}'] = df[f'x{i}'] - df['x0']
        df[f'y{i}'] = df[f'y{i}'] - df['y0']
        df[f'z{i}'] = df[f'z{i}'] - df['z0']
    
    coord_cols = [c for c in df.columns if any(x in c for x in ['x', 'y', 'z']) 
                  and c not in ['label', 'bpm', 'timestamp']]
    
    #Interpolacija korišćenjem timestampova jer kamera nije uvek 30fps
    timestamps = df['timestamp'].values
    relative_time = timestamps - timestamps[0]
    duration = relative_time[-1]
    new_time_steps = np.arange(0, duration, 1/TARGET_FPS)
    
    resampled_data = []
    for col in coord_cols:
        f = interp1d(relative_time, df[col].values, kind='linear', fill_value="extrapolate")
        resampled_data.append(f(new_time_steps))
    
    data_array = np.array(resampled_data).T
    label = LABEL_MAP[df['label'].iloc[0]]
    bpm = df['bpm'].iloc[0]

    #Računamo i brzinu (razlika između frejmova), dodajemo i to kao podatak modelu
    velocity = np.diff(data_array, axis=0)
    velocity = np.vstack([velocity[0], velocity])
    data_array = np.concatenate([data_array, velocity], axis=1)
    
    return data_array, label, bpm

#Funkcija da podelimo podatke iz csv-a na delove
def create_windows(data, label, bpm):
    X, y_t, y_b = [], [], []
    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        X.append(data[i : i + WINDOW_SIZE])
        y_t.append(label)
        y_b.append(bpm)
    return X, y_t, y_b

def load_train_val_sets(folder):
    X_train, y_t_train, y_b_train = [], [], []
    X_val, y_t_val, y_b_val = [], [], []
    
    files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    for file in files:
        data, label, bpm = preprocess_file(os.path.join(folder, file))
        
        #80/20 split u svakom fajlu iz dataset-a za train/val
        split_idx = int(len(data) * 0.8)
        train_chunk = data[:split_idx]
        val_chunk = data[split_idx:]
        
        xt, yt, yb = create_windows(train_chunk, label, bpm)
        X_train.extend(xt); y_t_train.extend(yt); y_b_train.extend(yb)
        
        xv, yv, yvb = create_windows(val_chunk, label, bpm)
        X_val.extend(xv); y_t_val.extend(yv); y_b_val.extend(yvb)
            
    return np.array(X_train), np.array(y_t_train), np.array(y_b_train), \
           np.array(X_val), np.array(y_t_val), np.array(y_b_val)

#Za testiranje koristimo poseban folder sa novim snimcima koji nisu u treningu
#Veličina test skupa je oko 1/3 veličine celog dataset skupa (train+val)
def load_test_set(folder):
    X_test, y_t_test, y_b_test = [], [], []
    
    if not os.path.exists(folder):
        print(f"Warning: {folder} not found!")
        return np.array([]), np.array([]), np.array([])

    files = sorted([f for f in os.listdir(folder) if f.endswith('.csv')])
    for file in files:
        data, label, bpm = preprocess_file(os.path.join(folder, file))
        xt, yt, yb = create_windows(data, label, bpm)
        X_test.extend(xt); y_t_test.extend(yt); y_b_test.extend(yb)
            
    return np.array(X_test), np.array(y_t_test), np.array(y_b_test)

#Učitavamo podatke
X_train, y_t_train, y_b_train, X_val, y_t_val, y_b_val = load_train_val_sets(DATA_FOLDER)
X_test, y_t_test, y_b_test = load_test_set(TEST_FOLDER)

print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

#Definicija modela
def build_conductor_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)

    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)  

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    
    out_task = layers.Dense(3, activation='softmax', name='task_out')(x)
    out_bpm = layers.Dense(1, activation='linear', name='bpm_out')(x)
    
    model = models.Model(inputs=inputs, outputs=[out_task, out_bpm])
    model.compile(
        optimizer='adam',
        loss={'task_out': 'sparse_categorical_crossentropy', 'bpm_out':  tf.keras.losses.Huber()},
        loss_weights={'task_out': 1.0, 'bpm_out': 0.05}, 
        metrics={'task_out': 'accuracy', 'bpm_out': 'mae'}
    )
    return model

model = build_conductor_model(X_train.shape[1:])

#Trening modela
history = model.fit(
    X_train, {'task_out': y_t_train, 'bpm_out': y_b_train},
    validation_data=(X_val, {'task_out': y_t_val, 'bpm_out': y_b_val}),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

#Testiranje i metrike
if len(X_test) > 0:
    pred_task_probs, pred_bpm = model.predict(X_test)
    pred_task = np.argmax(pred_task_probs, axis=1)

    print("\n--- TEST REZULTATI ---")
    print(classification_report(y_t_test, pred_task, target_names=["2_4", "3_4", "4_4"]))

    labels = ["2_4", "3_4", "4_4"]
    cm = confusion_matrix(y_t_test, pred_task)

    print("\nConfusion Matrix:")
    print("      Pred:", labels)
    for i, row in enumerate(cm):
        print(f"True {labels[i]}: {row}")

    print(f"BPM MAE: {mean_absolute_error(y_b_test, pred_bpm):.2f}")

#Čuvamo model
model.save('conductor_ai.h5')