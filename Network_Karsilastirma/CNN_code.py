# Gerekli kütüphanelerin import edilmesi
from tensorflow.keras.datasets import cifar10  # CIFAR-10 veri setini yüklemek için
from tensorflow.keras.models import Sequential  # Ardışık model yapısı oluşturmak için
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # CNN katmanları oluşturmak için gerekli katman tipleri
from tensorflow.keras.optimizers import Adam  # Optimizasyon algoritması olarak Adam'ı kullanmak için
from tensorflow.keras.utils import to_categorical  # Sınıf etiketlerini one-hot encoding yapmak için
from sklearn.metrics import classification_report, confusion_matrix  # Sınıflandırma raporu ve karmaşıklık matrisi oluşturmak için
from sklearn.model_selection import train_test_split  # Eğitim ve test veri setini bölmek için
import matplotlib.pyplot as plt  # Eğitim grafikleri çizmek için
import numpy as np  # Numpy kütüphanesi veri işlemlerinde kullanılır
from sklearn.preprocessing import StandardScaler  # Veriyi Z-score ile normalleştirmek için

# CIFAR-10 veri kümesinin yüklenmesi
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()  # CIFAR-10 veri seti eğitim ve test verisi olarak yükleniyor

# Eğitim ve test verilerini birleştirme
x_data = np.concatenate((x_train_full, x_test_full), axis=0)  # Eğitim ve test görüntü verisi birleştiriliyor
y_data = np.concatenate((y_train_full, y_test_full), axis=0)  # Eğitim ve test etiketleri birleştiriliyor

# Veri kümesini %70 eğitim ve %30 test olarak ayırma
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)  # %70 eğitim ve %30 test olarak bölünüyor

# Veriyi Z-score ile normalleştirme
scaler = StandardScaler()  # StandardScaler sınıfı kullanılarak Z-score normalizasyonu uygulanacak
# Veriyi 2 boyuta çevirip normalizasyon uygulanıyor, ardından orijinal boyutuna geri döndürülüyor
x_train = scaler.fit_transform(x_train.reshape(-1, 32 * 32 * 3)).reshape(-1, 32, 32, 3)
x_test = scaler.transform(x_test.reshape(-1, 32 * 32 * 3)).reshape(-1, 32, 32, 3)

# Sınıf etiketlerinin one-hot encoding yapılması
y_train = to_categorical(y_train, 10)  # Eğitim etiketleri 10 sınıfa göre one-hot encoding yapılıyor
y_test = to_categorical(y_test, 10)  # Test etiketleri 10 sınıfa göre one-hot encoding yapılıyor

# CNN Modelinin Tanımlanması
cnn_model = Sequential([  # Sequential modeli ile ardışık katmanlar oluşturuluyor
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),  # İlk evrişim katmanı, 32 filtre, 3x3 kernel, ReLU aktivasyonu ve 'same' padding
    MaxPooling2D((2, 2)),  # Max pooling katmanı, 2x2 penceresi ile boyut azaltma
    Conv2D(32, (3, 3), activation='relu', padding='same'),  # İkinci evrişim katmanı, aynı özelliklerle
    MaxPooling2D((2, 2)),  # İkinci max pooling katmanı
    Conv2D(32, (3, 3), activation='relu', padding='same'),  # Üçüncü evrişim katmanı, daha derin özellik haritaları çıkarılır
    Conv2D(32, (3, 3), activation='relu', padding='same'),  # Dördüncü evrişim katmanı
    Conv2D(32, (3, 3), activation='relu', padding='same'),  # Beşinci evrişim katmanı
    Flatten(),  # Düzleştirme katmanı, 2D veriyi 1D vektöre dönüştürür
    Dense(128, activation='relu'),  # Tam bağlantılı katman: 128 nöron, ReLU aktivasyonu
    Dense(128, activation='relu'),  # İkinci tam bağlantılı katman, aynı özelliklerle
    Dense(10, activation='softmax')  # Çıkış katmanı: 10 sınıf için softmax aktivasyonu ile sınıf tahmini yapılıyor
])

# CNN modelini derleme
cnn_model.compile(optimizer=Adam(learning_rate=0.0004),  # Adam optimizasyon algoritması seçildi ve öğrenme oranı 0.0004 olarak ayarlandı
                  loss='categorical_crossentropy',  # Çok sınıflı sınıflandırma için kategorik çapraz entropi kaybı
                  metrics=['accuracy'])  # Modelin başarımını doğrulamak için doğruluk metriği kullanılıyor

print("CNN Model Özeti:")
cnn_model.summary()  # Bu komut modelin yapısını ve parametre sayısını gösterir

# CNN modelini eğitme
cnn_history = cnn_model.fit(x_train, y_train, epochs=5, batch_size=8, validation_split=0.1)
# Model 5 epoch boyunca eğitiliyor, batch boyutu 8 olarak ayarlanmış ve doğrulama verisi olarak eğitim verisinin %10'u ayrılıyor

# CNN model metrikleri
cnn_predictions = np.argmax(cnn_model.predict(x_test), axis=1)  # Tahmin edilen sınıf etiketleri (en yüksek olasılığa sahip sınıf) alınıyor
y_true = np.argmax(y_test, axis=1)  # Gerçek etiketler (one-hot encoded formatından çıkarılıyor)
cnn_conf_matrix = confusion_matrix(y_true, cnn_predictions)  # Karmaşıklık matrisi hesaplanıyor
cnn_class_report = classification_report(y_true, cnn_predictions)  # Sınıflandırma raporu oluşturuluyor

print("CNN Confusion Matrix:\n", cnn_conf_matrix)  # Karmaşıklık matrisi çıktı olarak gösteriliyor
print("CNN Classification Report:\n", cnn_class_report)  # Sınıflandırma raporu çıktı olarak gösteriliyor

# Eğitim grafikleri - CNN
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='CNN Training Accuracy')  # Eğitim doğruluğu çiziliyor
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')  # Doğrulama doğruluğu çiziliyor
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='CNN Training Loss')  # Eğitim kaybı çiziliyor
plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss')  # Doğrulama kaybı çiziliyor
plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()  # Eğitim doğruluk ve kayıp grafikleri gösteriliyor

with open('CNN metrik_sonuclari.txt', 'w') as f:
    f.write("CNN Model Classification Report:\n")
    f.write(cnn_class_report)
    f.write("\nCNN Model Confusion Matrix:\n")
    f.write(str(cnn_conf_matrix))

# Ağırlıkları kaydetme
cnn_model.save_weights('cnn_weights.h5')  # Eğitilen modelin ağırlıkları bir dosyaya kaydediliyor

