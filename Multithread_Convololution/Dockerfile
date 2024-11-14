# Ubuntu tabanlı bir Docker imajı başlatın
FROM ubuntu:latest

# Gerekli bağımlılıkları yükleyin ve sistemi güncelleyin
RUN apt update && \
    apt install -y g++ libomp-dev libopencv-dev pkg-config

# Proje dosyalarını konteynıra kopyalayın
COPY conv_omp.cpp /root/conv_omp.cpp
COPY soru_3_gorsel.png /root/soru_3_gorsel.png

# Kodu OpenMP ve OpenCV desteğiyle derleyin
RUN g++ -fopenmp /root/conv_omp.cpp -o /root/conv_omp `pkg-config --cflags --libs opencv4`

# Çalıştırılabilir dosyayı başlatın ve konteyneri açık tutun
CMD ["/bin/bash", "-c", "/root/conv_omp; tail -f /dev/null"]
