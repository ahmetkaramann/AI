#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <omp.h>

#define KERNEL_SIZE 5

// 5x5 Gauss bulanıklaştırma çekirdeği
int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1, 4, 7, 4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1, 4, 7, 4, 1}
};
const int kernelWeight = 273;  // Kernel için normalizasyon faktörü

// Belirli bir satır aralığı için konvolüsyon fonksiyonu
void applyConvolution(const cv::Mat& input, cv::Mat& output, int startRow, int endRow, std::vector<int>& histogram) {
    // Belirtilen satır aralığındaki her piksel için işlem yapılacak
    for (int x = startRow; x < endRow; x++) {
        for (int y = 2; y < input.cols - 2; y++) {  // Piksel kenar sınırları nedeniyle 2 piksel içeriden başlanır
            int sum = 0;

            // 5x5 çekirdekle konvolüsyon işlemi gerçekleştir
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    sum += input.at<uchar>(x + i, y + j) * kernel[i + 2][j + 2]; // Çekirdekteki ağırlık ile piksel değeri çarpılarak toplanır
                }
            }
            int pixelValue = std::clamp(sum / kernelWeight, 0, 255); // Piksel değeri normalize edilerek sınırlandırılır
            output.at<uchar>(x, y) = pixelValue; // Output için yeni piksel değeri atanır

            // Histogram thread-safe bir şekilde güncellenir (atomatic işlem)
            #pragma omp atomic
            histogram[pixelValue]++;
        }
    }
}

int main() {

    // Input görseli içe aktarma
    cv::Mat image = cv::imread("/root/soru_3_gorsel.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Görsel yüklenemedi!" << std::endl;
        return -1;
    }

    // Output görüntüsü oluşturma (aynı boyutlarda sıfır matrisi)
    cv::Mat output = cv::Mat::zeros(image.size(), CV_8U);
    int numThreads = 5; // Kullanılacak iş parçacığı sayısı
    int rowsPerThread = (image.rows - 4) / numThreads; // Her iş parçacığına düşen satır sayısı

    // Histogramı sıfırla (0 ile 255 arasındaki indeksler)
    std::vector<int> histogram(256, 0);

    // Paralel işlem için süre ölçümü başlat
    double startParallel = omp_get_wtime();
    
    #pragma omp parallel num_threads(numThreads) 
    {
        int thread_id = omp_get_thread_num(); // Her iş parçacığı kendine özel bir 'thread_id' alır
        int startRow = 2 + thread_id * rowsPerThread; // İş parçacığının işlem yapacağı başlangıç satırı hesaplanır
        int endRow = (thread_id == numThreads - 1) ? image.rows - 2 : startRow + rowsPerThread; // İş parçacığının işlem yapacağı bitiş satırı belirlenir, son iş parçacığına kalan satırlar atanır
    applyConvolution(image, output, startRow, endRow, histogram); 
    }

        

    // Paralel işlem için süre ölçümü bitir ve geçen süreyi hesapla
    double endParallel = omp_get_wtime();
    double parallelTime = endParallel - startParallel;

    // Tek iş parçacıklı çalışmayı sıfırla 
    histogram.assign(256, 0);
    output = cv::Mat::zeros(image.size(), CV_8U);

    // Tek iş parçacıklı işlem için süre ölçümü başlat
    double startSingle = omp_get_wtime();

    applyConvolution(image, output, 2, image.rows - 2, histogram);

    // Tek iş parçacıklı işlem için süre ölçümü bitir ve geçen süreyi hesapla
    double endSingle = omp_get_wtime();
    double singleTime = endSingle - startSingle;

    // Hızlanma oranını hesapla
    double speedup = singleTime / parallelTime;

    // Ölçümleri loga yazdırma
    std::cout << "Tek iş parçacıklı süre: " << singleTime << " saniye" << std::endl;
    std::cout << "Paralel süre: " << parallelTime << " saniye" << std::endl;
    std::cout << "Hızlanma: " << speedup << "x" << std::endl;

    // Histogram çizimi ve kaydetme kısmı
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w / 256);
    cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(50, 50, 50)); // Daha koyu gri arka plan

    // Histogram değerlerini görüntü yüksekliğine uyarlamak için normalize et
    int max_value = *std::max_element(histogram.begin(), histogram.end());
    for (int i = 0; i < 256; i++) {
        histogram[i] = ((double)histogram[i] / max_value) * histImage.rows;
    }

    // Histogramı çubuklar (dikdörtgenler) olarak çiz
    for (int i = 0; i < 256; i++) {
        cv::rectangle(histImage,
                      cv::Point(bin_w * i, hist_h - 1),
                      cv::Point(bin_w * i + bin_w, hist_h - histogram[i]),
                      cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // X ve Y eksenlerini çiz
    cv::line(histImage, cv::Point(0, hist_h - 1), cv::Point(hist_w, hist_h - 1), cv::Scalar(255, 255, 255), 2); // X ekseni
    cv::line(histImage, cv::Point(0, 0), cv::Point(0, hist_h), cv::Scalar(255, 255, 255), 2); // Y ekseni

    // X ve Y eksenlerine etiketler ekle (0 ve 255. pointler dahil)
    cv::putText(histImage, "Piksel Yogunlugu", cv::Point(hist_w / 2 - 50, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(histImage, "Frekans", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(histImage, "0", cv::Point(5, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(histImage, "255", cv::Point(hist_w - 30, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Histogramı kaydet
    cv::imwrite("/root/histogram.png", histImage);
    std::cout << "Çubuk grafik şeklindeki histogram 'histogram.png' olarak kaydedildi." << std::endl;

    // Çıktı görüntüsünü kaydet
    cv::imwrite("/root/output.png", output);
    std::cout << "Output görüntü 'output.png' olarak kaydedildi." << std::endl;

    return 0;
}
