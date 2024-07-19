#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

const int NUM_THREADS = std::thread::hardware_concurrency();
const int ZOOM_FACTOR = 4;
const int TARGET_WIDTH = 1280;
const int TARGET_HEIGHT = 960;

inline cv::Vec3b interpolate(const cv::Mat& img, float x, float y) {
    static const float weights[16] = {
        -0.041026f, -0.016225f, -0.016225f, -0.041026f,
        -0.016225f,  0.323475f,  0.323475f, -0.016225f,
        -0.016225f,  0.323475f,  0.323475f, -0.016225f,
        -0.041026f, -0.016225f, -0.016225f, -0.041026f
    };
    float sum[3] = {0, 0, 0};
    int baseX = static_cast<int>(x) - 1;
    int baseY = static_cast<int>(y) - 1;
    for (int j = 0; j < 4; ++j) {
        int yi = cv::borderInterpolate(baseY + j, img.rows, cv::BORDER_REFLECT_101);
        for (int i = 0; i < 4; ++i) {
            int xi = cv::borderInterpolate(baseX + i, img.cols, cv::BORDER_REFLECT_101);
            const cv::Vec3b& pixel = img.at<cv::Vec3b>(yi, xi);
            float weight = weights[j * 4 + i];
            sum[0] += weight * pixel[0];
            sum[1] += weight * pixel[1];
            sum[2] += weight * pixel[2];
        }
    }
    return cv::Vec3b(cv::saturate_cast<uchar>(sum[0]),
                     cv::saturate_cast<uchar>(sum[1]),
                     cv::saturate_cast<uchar>(sum[2]));
}

cv::Mat digitalZoom(const cv::Mat& input) {
    cv::Mat resizedInput;
    cv::resize(input, resizedInput, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));

    int croppedSize = std::min(resizedInput.cols, resizedInput.rows) / ZOOM_FACTOR;
    int startX = (resizedInput.cols - croppedSize) / 2;
    int startY = (resizedInput.rows - croppedSize) / 2;
    
    cv::Mat cropped = resizedInput(cv::Rect(startX, startY, croppedSize, croppedSize));
    cv::Mat output(TARGET_HEIGHT, TARGET_WIDTH, input.type());

    auto processRows = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < output.cols; ++x) {
                float srcX = (x * croppedSize) / static_cast<float>(output.cols);
                float srcY = (y * croppedSize) / static_cast<float>(output.rows);
                output.at<cv::Vec3b>(y, x) = interpolate(cropped, srcX, srcY);
            }
        }
    };

    std::vector<std::thread> threads;
    int rowsPerThread = output.rows / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; ++t) {
        int startY = t * rowsPerThread;
        int endY = (t == NUM_THREADS - 1) ? output.rows : startY + rowsPerThread;
        threads.emplace_back(processRows, startY, endY);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return output;
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cv::Mat frame, zoomedFrame;
    cv::namedWindow("Original Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Zoomed Video", cv::WINDOW_NORMAL);

    while (true) {


        cap >> frame;
        if (frame.empty()) break;
        cv::resize(frame, frame, cv::Size(TARGET_WIDTH, TARGET_HEIGHT));


        cv::imshow("Original Video", frame);

        auto processingStart = std::chrono::high_resolution_clock::now();


        zoomedFrame = digitalZoom(frame);

        auto processingEnd = std::chrono::high_resolution_clock::now();



        cv::imshow("Zoomed Video", zoomedFrame);



        auto processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(processingEnd - processingStart);

        std::cout   << "Processing time: " << processingTime.count() << " ms" << std::endl;

        if (cv::waitKey(1) == 27) break;  // ESC key
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
