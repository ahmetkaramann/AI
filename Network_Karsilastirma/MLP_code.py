# Gerekli kütüphanelerin import edilmesi
from tensorflow.keras.datasets import cifar10  # CIFAR-10 veri setini yüklemek için
from tensorflow.keras.models import Sequential  # Ardışık model yapısı oluşturmak için
from tensorflow.keras.layers import Dense, Flatten  # Katman tiplerini belirlemek için
from tensorflow.keras.optimizers import Adam  # Optimizasyon algoritması olarak Adam'ı kullanmak için
from tensorflow.keras.utils import to_categorical  # Sınıf etiketlerini one-hot encoding yapmak için
from sklearn.metrics import classification_report, confusion_matrix  # Sınıflandırma raporu ve karmaşıklık matrisi oluşturmak için
from sklearn.model_selection import train_test_split  # Eğitim ve test veri setini bölmek için
import matplotlib.pyplot as plt  # Eğitim grafikleri çizmek için
import numpy as np  #  veri işlemlerinde kullanılır
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

# MLP Modelinin Tanımlanması
mlp_model = Sequential([  # Sequential modeli ile ardışık katmanlar oluşturuluyor
    Flatten(input_shape=(32, 32, 3)),  # 32x32x3 boyutundaki giriş verisi tek bir vektöre dönüştürülüyor
    Dense(128, activation='relu'),  # İkinci katman: 128 nöronlu ve ReLU aktivasyon fonksiyonlu Dense katmanı
    Dense(128, activation='relu'),  # Üçüncü katman: 128 nöronlu ve ReLU aktivasyon fonksiyonlu Dense katmanı
    Dense(128, activation='relu'),  # Dördüncü katman: 128 nöronlu ve ReLU aktivasyon fonksiyonlu Dense katmanı
    Dense(128, activation='relu'),  # Beşinci katman: 128 nöronlu ve ReLU aktivasyon fonksiyonlu Dense katmanı
    Dense(128, activation='relu'),  # Altıncı katman: 128 nöronlu ve ReLU aktivasyon fonksiyonlu Dense katmanı
    Dense(10, activation='softmax')  # Çıkış katmanı: 10 sınıf için softmax aktivasyonu ile sınıf tahmini yapılıyor
])

# MLP modelini derleme
mlp_model.compile(optimizer=Adam(learning_rate=0.0001),  # Adam optimizasyon algoritması seçildi ve öğrenme oranı 0.0001 olarak ayarlandı
                  loss='categorical_crossentropy',  # Çok sınıflı sınıflandırma için kategorik çapraz entropi kaybı
                  metrics=['accuracy'])  # Modelin başarımını doğrulamak için doğruluk metriği kullanılıyor

print("MLP Model Özeti:")
mlp_model.summary()  # Bu komut modelin yapısını ve parametre sayısını gösterir

# MLP modelini eğitme
mlp_history = mlp_model.fit(x_train, y_train, epochs=5, batch_size=8, validation_split=0.1)
# Model 5 epoch boyunca eğitiliyor, batch boyutu 8 olarak ayarlanmış ve doğrulama verisi olarak eğitim verisinin %10'u ayrılıyor

# MLP model metrikleri
mlp_predictions = np.argmax(mlp_model.predict(x_test), axis=1)  # Tahmin edilen sınıf etiketleri (en yüksek olasılığa sahip sınıf) alınıyor
y_true = np.argmax(y_test, axis=1)  # Gerçek etiketler (one-hot encoded formatından çıkarılıyor)
mlp_conf_matrix = confusion_matrix(y_true, mlp_predictions)  # Karmaşıklık matrisi hesaplanıyor
mlp_class_report = classification_report(y_true, mlp_predictions)  # Sınıflandırma raporu oluşturuluyor

print("MLP Confusion Matrix:\n", mlp_conf_matrix)  # Karmaşıklık matrisi çıktı olarak gösteriliyor
print("MLP Classification Report:\n", mlp_class_report)  # Sınıflandırma raporu çıktı olarak gösteriliyor

# Eğitim grafikleri - MLP
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'], label='MLP Training Accuracy')  # Eğitim doğruluğu çiziliyor
plt.plot(mlp_history.history['val_accuracy'], label='MLP Validation Accuracy')  # Doğrulama doğruluğu çiziliyor
plt.title('MLP Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mlp_history.history['loss'], label='MLP Training Loss')  # Eğitim kaybı çiziliyor
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')  # Doğrulama kaybı çiziliyor
plt.title('MLP Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()  # Eğitim doğruluk ve kayıp grafikleri gösteriliyor

with open('metrik_sonuclari.txt', 'w') as f:
    f.write("MLP Model Classification Report:\n")
    f.write(mlp_class_report)
    f.write("\nMLP Model Confusion Matrix:\n")
    f.write(str(mlp_conf_matrix))

# Ağırlıkları kaydetme
mlp_model.save_weights('mlp_weights.h5')  # Eğitilen modelin ağırlıkları bir dosyaya kaydediliyor

