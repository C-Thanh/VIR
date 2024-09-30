#include "FeatureExtractor.hpp"

//Color Histogram
Mat ColorHistogramExtractor::extractFeature(const Mat& image) {
    std::vector<Mat> channels;
    cv::split(image, channels);

    int histSize = 256; // Number of binsA
    float range[] = { 0, 256 }; // Range of values
    const float* histRange = { range };

    Mat hist;
    std::vector<Mat> histograms;

    for (int i = 0; i < 3; ++i) {
        Mat channel_hist;
        cv::calcHist(&channels[i], 1, 0, Mat(), channel_hist, 1, &histSize, &histRange);
        histograms.push_back(channel_hist);
    }

    cv::hconcat(histograms, hist);
    hist /= cv::sum(hist)[0];
    return hist.reshape(1, 1); // Flatten histogram
}

//Color Correlogram
Mat ColorCorrelogramExtractor::extractFeature(const Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    // Quantize the image colors
    Mat quantizedImage;
    quantizeColors(image, quantizedImage);

    // Calculate the correlogram
    std::vector<float> correlogram;
    calculateCorrelogram(quantizedImage, correlogram);

    // Convert correlogram vector to Mat and reshape to single row
    return Mat(correlogram).reshape(1, 1);
}

void ColorCorrelogramExtractor::quantizeColors(const Mat& src, Mat& dst, int colorLevels) {
    dst = src.clone();
    int step = 256 / colorLevels;
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            for (int c = 0; c < src.channels(); ++c) {
                dst.at<cv::Vec3b>(y, x)[c] = src.at<cv::Vec3b>(y, x)[c] / step * step + step / 2;
            }
        }
    }
}

void ColorCorrelogramExtractor::calculateCorrelogram(const Mat& image, std::vector<float>& correlogram, int maxDistance) {
    int colorLevels = 8;
    int numBins = colorLevels * colorLevels * colorLevels;
    correlogram.assign(numBins * maxDistance, 0);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            int colorIdx = (color[0] / 32) * 64 + (color[1] / 32) * 8 + (color[2] / 32);

            for (int d = 1; d <= maxDistance; ++d) {
                int count = 0;
                for (int dy = -d; dy <= d; ++dy) {
                    for (int dx = -d; dx <= d; ++dx) {
                        if (dy == 0 && dx == 0) continue;
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < image.rows && nx >= 0 && nx < image.cols) {
                            cv::Vec3b neighborColor = image.at<cv::Vec3b>(ny, nx);
                            int neighborColorIdx = (neighborColor[0] / 32) * 64 + (neighborColor[1] / 32) * 8 + (neighborColor[2] / 32);
                            if (colorIdx == neighborColorIdx) {
                                ++count;
                            }
                        }
                    }
                }
                correlogram[colorIdx * maxDistance + d - 1] += static_cast<float>(count);
            }
        }
    }

    // Normalize the correlogram
    for (size_t i = 0; i < correlogram.size(); ++i) {
        correlogram[i] /= (image.rows * image.cols);
    }
}
