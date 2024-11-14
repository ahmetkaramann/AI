import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görselleri yükleyin
image1_path = 'soru_2_gorsel1.png'
image2_path = 'soru_2_gorsel2.png'
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Görüntüleri gri tonlamaya çevirin ve kontrastı artırmak için CLAHE uygulayın
# CLAHE, kontrastı dengeleyerek düşük kontrastlı alanlarda daha iyi anahtar nokta tespiti sağlar
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray1 = clahe.apply(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY))
gray2 = clahe.apply(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))

# SIFT ile anahtar noktaları ve tanımlayıcıları bulun
# SIFT parametreleri (edgeThreshold ve contrastThreshold) optimize edilmiştir
# edgeThreshold, zayıf kenarları kaldırır, contrastThreshold ise düşük kontrastlı anahtar noktaları eler
sift = cv2.SIFT_create(edgeThreshold=10, contrastThreshold=0.04)
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# BFMatcher ve Lowe'un oran testi ile eşleşmeleri bulun
# BFMatcher ile anahtar noktalar arasındaki mesafeye göre eşleşmeler yapılır
bf = cv2.BFMatcher(cv2.NORM_L2)

# Lowe'un oran testi kullanılarak yanlış eşleşmeler filtrelenir; en iyi iki eşleşme arasındaki mesafe kıyaslanır
# Oran testinde, ilk eşleşme mesafesi, ikinciye göre yeterince küçükse "iyi eşleşme" olarak kabul edilir
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # Oran testi eşiği (0.75) optimize edilmiştir
        good_matches.append(m)

# Eğer yeterli sayıda iyi eşleşme varsa, homografi matrisini hesapla ve panoramik görüntüyü oluştur
if len(good_matches) >= 4:
    # Kaynak (src) ve hedef (dst) noktalarını elde et
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC algoritmasıyla homografi matrisini hesapla; bu matris iki görüntü arasındaki dönüşümü tanımlar
    # RANSAC, yanlış eşleşmeleri dışlayarak dönüşüm matrisinin doğruluğunu artırır
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Eşleşmeleri görselleştir
    # Eşleşmeler, yeşil renkle çizilir ve yalnızca doğru eşleşmeler gösterilir
    matches_mask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matches_mask, flags=2)
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, **draw_params)

    # Eşleşme görselini göster
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title(f"İyileştirilmiş Eşleşmeler : {len(good_matches)}")
    plt.axis("off")
    plt.show()

    # Homografi matrisini kullanarak ikinci görüntüyü birleştir
    # Birinci görüntüye ikinci görüntüyü eklemek için cv2.warpPerspective ile dönüşüm uygulanır
    height, width = image1.shape[:2]
    result = cv2.warpPerspective(image2, H, (width * 2, height))
    result[0:height, 0:width] = image1  # İlk görüntüyü sonucun sol tarafına yerleştir

    # Panoramik görüntüyü iyileştirmek için siyah kenarları kaldırma işlemi yapılır
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # Gri tonlama
    _, thresh = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)  # Siyah bölgeleri bulmak için eşikleme uygula

    # Siyah kenarları kırpmak için konturları bulun
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    result_cropped = result[y:y+h, x:x+w]  # Kırpılmış panoramik görüntü

    # Panoramik görüntüyü göster
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(result_cropped, cv2.COLOR_BGR2RGB))
    plt.title("Homografi ile Oluşturulan Panoramik Görüntü")
    plt.axis("off")
    plt.show()

    # İyi eşleşme sayısını ve homografi matrisini yazdır
    print(f"Geliştirilmiş İyi Eşleşmeler: {len(good_matches)}")
    print("Geliştirilmiş Homografi Matrisi (H):")
    print(H)
else:
    # Yeterli eşleşme bulunamazsa uyarı ver
    print("Yeterli eşleşme bulunamadı.")
