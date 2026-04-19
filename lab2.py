import os
import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks, backend as K
from gtts import gTTS
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
model_path = 'speech_recognition_model.keras'
DATASET_PATH = "dataset/"
SAMPLE_RATE = 16000
N_MFCC = 20
MAX_PAD_LEN = 52

word_to_char = {
    'alpha': 'A', 'bravo': 'B', 'charlie': 'C', 'delta': 'D', 'echo': 'E',
    'foxtrot': 'F', 'golf': 'G', 'hotel': 'H', 'india': 'I', 'juliett': 'J',
    'kilo': 'K', 'lima': 'L', 'mike': 'M', 'november': 'N', 'oscar': 'O',
    'papa': 'P', 'quebec': 'Q', 'romeo': 'R', 'sierra': 'S', 'tango': 'T',
    'uniform': 'U', 'victor': 'V', 'whiskey': 'W', 'xray': 'X', 'yankee': 'Y', 'zulu': 'Z'
}
class_names = sorted(list(word_to_char.keys()))


def extract_features(audio, sr):
    audio = librosa.util.normalize(audio)
    audio = librosa.effects.preemphasis(audio, coef=0.98)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    combined = np.vstack([mfcc, delta, rolloff])
    combined = (combined - np.mean(combined, axis=1, keepdims=True)) / (np.std(combined, axis=1, keepdims=True) + 1e-8)
    
    if combined.shape[1] < MAX_PAD_LEN:
        pad = MAX_PAD_LEN - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (pad // 2, pad - pad // 2)), mode='constant')
    else:
        combined = combined[:, :MAX_PAD_LEN]
    return combined

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_acc'], label='Навчання (Mixup)', color='#3498db', lw=2)
    ax1.plot(history['val_acc'], label='Валідація', color='#e67e22', lw=2)
    ax1.set_title('Точність моделі (Accuracy)')
    ax1.set_xlabel('Епоха')
    ax1.set_ylabel('Точність')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(history['train_loss'], label='Навчання (Mixup)', color='#3498db', lw=2)
    ax2.plot(history['val_loss'], label='Валідація', color='#e67e22', lw=2)
    ax2.set_title('Функція втрат (Loss)')
    ax2.set_xlabel('Епоха')
    ax2.set_ylabel('Втрати')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    return mixed_x, mixed_y


if not os.path.exists(model_path):
    print(f"\n[INFO] Генерація датасету та навчання моделі...")
    
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)
        accents = ['com', 'co.uk', 'com.au', 'ca', 'co.in', 'ie', 'co.za', 'com.ng', 'com.ph', 'com.gh']
        for word in class_names:
            word_dir = os.path.join(DATASET_PATH, word)
            os.makedirs(word_dir, exist_ok=True)
            for tld in accents:
                try: gTTS(text=word, lang='en', tld=tld).save(os.path.join(word_dir, f"{word}_{tld}.mp3"))
                except: continue

    X, Y = [], []
    for idx, name in enumerate(class_names):
        d = os.path.join(DATASET_PATH, name)
        for f in os.listdir(d):
            if f.endswith('.mp3'):
                audio, _ = librosa.load(os.path.join(d, f), sr=SAMPLE_RATE)
                audio, _ = librosa.effects.trim(audio, top_db=20)
                feat = extract_features(audio, SAMPLE_RATE)
                X.append(feat); Y.append(idx)

    X = np.array(X).reshape(-1, N_MFCC * 2 + 1, MAX_PAD_LEN, 1)
    Y = tf.keras.utils.to_categorical(np.array(Y), len(class_names))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

    model = models.Sequential([
        layers.Input(shape=(N_MFCC * 2 + 1, MAX_PAD_LEN, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history_dict = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

    print("\n[TRAINING] Початок навчання (60 епох)...")
    for epoch in range(60):
        m_x, m_y = mixup_data(X_train, y_train)
        h = model.fit(m_x, m_y, batch_size=32, verbose=0)
        
        val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
        
        history_dict['train_acc'].append(h.history['accuracy'][0])
        history_dict['train_loss'].append(h.history['loss'][0])
        history_dict['val_acc'].append(val_acc)
        history_dict['val_loss'].append(val_loss)

        if epoch % 10 == 0 or epoch == 59:
            print(f"Епоха {epoch:2d}: loss={h.history['loss'][0]:.4f}, acc={h.history['accuracy'][0]:.4f} | val_acc={val_acc:.4f}")

    model.save(model_path)
    print("\n[INFO] Модель збережена. Будуємо графік навчання...")
    plot_training_history(history_dict)
else:
    print(f"[INFO] Завантаження існуючої моделі: {model_path}")
    model = models.load_model(model_path)


def predict_custom(path, model):
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)
        y_trim, _ = librosa.effects.trim(y, top_db=25)
        intervals = librosa.effects.split(y_trim, top_db=25, frame_length=2048)
        
        print("\n" + "═"*50)
        print(f"АНАЛІЗ ФАЙЛУ: {os.path.basename(path)}")
        print("═"*50)

        for i, (s, e) in enumerate(intervals):
            chunk = y_trim[s:e]
            if len(chunk) < sr * 0.15: continue 
            
            feat = extract_features(chunk, sr).reshape(1, N_MFCC * 2 + 1, MAX_PAD_LEN, 1)
            pred = model.predict(feat, verbose=0)[0]
            
            top_5_indices = np.argsort(pred)[-5:][::-1]
            top_5_words = [class_names[idx].upper() for idx in top_5_indices]
            top_5_probs = [pred[idx] * 100 for idx in top_5_indices]
            
            plt.figure(figsize=(8, 4))
            colors = ['#2ecc71' if x == max(top_5_probs) else '#3498db' for x in top_5_probs]
            bars = plt.barh(top_5_words[::-1], top_5_probs[::-1], color=colors[::-1])
            plt.title(f'Слово #{i+1}: Ймовірна команда - {top_5_words[0]}')
            plt.xlabel('Впевненість (%)')
            plt.xlim(0, 105)
            
            for bar in bars:
                plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                         f'{bar.get_width():.1f}%', va='center')
            plt.tight_layout()
            plt.show()
            
            print(f" Слово #{i+1}: {top_5_words[0]:<10} | Точність: {top_5_probs[0]:>5.2f}%")
            
        print("═"*50 + "\n")
    except Exception as e:
        print(f"Помилка при обробці: {e}")


while True:
    cmd = input("Натисніть Enter щоб вибрати файл (або 'q' для виходу): ")
    if cmd.lower() == 'q': break
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    p = filedialog.askopenfilename(title="Виберіть аудіофайл", filetypes=[("Audio files", "*.mp3 *.wav *.m4a")])
    root.destroy()
    
    if p:
        predict_custom(p, model)