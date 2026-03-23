import os
import cv2
import numpy as np
import string
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split

symbols = string.ascii_lowercase + string.digits
num_classes = len(symbols)
char_to_num = {char: i for i, char in enumerate(symbols)}
num_to_char = {i: char for i, char in enumerate(symbols)}

model_path = 'captcha_model.keras'

if os.path.exists(model_path):
    print(f"\n[INFO] Знайдено збережену модель '{model_path}'. Завантажуємо...")
    model = models.load_model(model_path)
    print("[УСПІХ] Модель успішно завантажено! Пропускаємо етап тренування.\n")
    
else:
    print(f"\n[INFO] Файл '{model_path}' не знайдено. Починаємо завантаження даних та тренування...\n")
    
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_b894c142ba413bed91949aeee4e95b10'
    extract_dir = 'captcha_data'

    if not os.path.exists(extract_dir):
        print("Downloading archive from Kaggle... (this may take a minute)")
        import kaggle
        kaggle.api.dataset_download_files('parsasam/captcha-dataset', path=extract_dir, unzip=True)
        print("Data successfully downloaded and extracted!\n")

    X = []
    Y = []
    
    data_dir = extract_dir
    for root, dirs, files in os.walk(extract_dir):
        if any(f.endswith('.png') or f.endswith('.jpg') for f in files):
            data_dir = root
            break

    for filename in os.listdir(data_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            label = filename.split('.')[0].lower()        
            if len(label) == 5:
                img_path = os.path.join(data_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img = cv2.resize(img, (200, 50))
                    img = img / 255.0
                    X.append(img)
                    Y.append([char_to_num[char] for char in label])

    X = np.array(X).reshape(-1, 50, 200, 1)
    Y = np.array(Y)
    
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)

    input_layer = layers.Input(shape=(50, 200, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(5 * num_classes)(x)
    x = layers.Reshape((5, num_classes))(x)
    output_layer = layers.Activation('softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=40, batch_size=16, validation_data=(X_val, Y_val), verbose=1)

    model.save(model_path)
    print(f"\n[УСПІХ] Нову модель збережено у файл: {model_path}\n")


def predict_custom_image(image_path, model, num_to_char):
    if not os.path.exists(image_path):
        print(f"Помилка: Файл '{image_path}' не знайдено.")
        return

    img_array = np.fromfile(image_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Помилка читання файлу.")
        return

    img_resized = cv2.resize(img, (200, 50))
    img_normalized = img_resized / 255.0
    img_input = np.array(img_normalized).reshape(-1, 50, 200, 1)

    preds = model.predict(img_input, verbose=0)
    
    predicted_text = ""
    for j in range(5):
        pred_char_index = np.argmax(preds[0][j])
        predicted_text += num_to_char[pred_char_index]

    print(f"\n--> Розпізнаний текст: {predicted_text} <--\n")
    
    plt.figure(figsize=(6, 3))
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted: {predicted_text}", color='blue', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Оберіть картинку з капчею",
        filetypes=[("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
    )
    
    root.destroy()
    return file_path

while True:
    user_input = input("Натисніть Enter, щоб обрати картинку (або введіть 'q' для виходу): ")
    
    if user_input.lower() == 'q':
        print("Роботу завершено. Гарного дня!")
        break
        
    selected_file = select_image_file()
    
    if not selected_file:
        print("Файл не обрано. Спробуйте ще раз.")
        continue
        
    predict_custom_image(selected_file, model, num_to_char)