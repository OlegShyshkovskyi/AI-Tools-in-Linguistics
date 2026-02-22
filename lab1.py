import os
import cv2
import numpy as np
import string
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_b894c142ba413bed91949aeee4e95b10'

extract_dir = 'captcha_data'

if not os.path.exists(extract_dir):
    print("Downloading archive from Kaggle... (this may take a minute)")
    try:
        import kaggle
        kaggle.api.dataset_download_files('fournierp/captcha-version-2-images', path=extract_dir, unzip=True)
        print("Data successfully downloaded and extracted!\n")
    except Exception as e:
        print(f"Error connecting to Kaggle: {e}")
        print("Make sure you have installed the Kaggle library (pip install kaggle in the terminal)")
        exit()
else:
    print("Data folder already exists, proceeding to processing.\n")

symbols = string.ascii_lowercase + string.digits
num_classes = len(symbols)

char_to_num = {char: i for i, char in enumerate(symbols)}
num_to_char = {i: char for i, char in enumerate(symbols)}

X = []
Y = []

data_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if any(f.endswith('.png') or f.endswith('.jpg') for f in files):
        data_dir = root
        break

if not os.path.exists(data_dir) or len(os.listdir(data_dir)) == 0:
    print(f"Critical error: Folder {data_dir} is empty or does not exist. Delete the captcha_data folder and run again.")
    exit()

for filename in os.listdir(data_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        label = filename.split('.')[0]
        
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

print(f"Successfully processed {len(X)} images.")

X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)

print(f"Sample sizes -> Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}\n")


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

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(
    X_train, Y_train,
    epochs=20, 
    batch_size=2,
    validation_data=(X_val, Y_val),
    verbose=1
)

print("\nQuality assessment on the test sample (data the network hasn't seen):")
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"Average accuracy for recognizing characters: {acc * 100:.2f}%\n")

print("Generating training plots...")
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
line1 = ax1.plot(history.history['loss'], label='Training Loss', color='lightcoral', linestyle='--')
line2 = ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True)

ax2 = ax1.twinx()  
ax2.set_ylabel('Accuracy', color='tab:blue')
line3 = ax2.plot(history.history['accuracy'], label='Training Accuracy', color='lightskyblue', linestyle='--')
line4 = ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue', linewidth=2)
ax2.tick_params(axis='y', labelcolor='tab:blue')

lines = line1 + line2 + line3 + line4
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Training Loss and Average Accuracy')
fig.tight_layout()
plt.show()

preds = model.predict(X_test[:5])

plt.figure(figsize=(15, 5))
for i in range(5):
    predicted_text = ""
    for j in range(5):
        pred_char_index = np.argmax(preds[i][j])
        predicted_text += num_to_char[pred_char_index]
    
    true_text = "".join([num_to_char[Y_test[i][j]] for j in range(5)])
    
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i].reshape(50, 200), cmap='gray')
    plt.title(f"True: {true_text}\nPredicted: {predicted_text}", 
              color='green' if true_text == predicted_text else 'red',
              fontsize=12, fontweight='bold')
    plt.axis('off')

plt.tight_layout()
plt.show()