#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

using namespace cv;

class FeatureExtractorInterface {
public:
	virtual Mat extractFeature(const Mat& image) = 0;
	virtual ~FeatureExtractorInterface() {}
};

class ColorHistogramExtractor : public FeatureExtractorInterface {
public:
	Mat extractFeature(const Mat& image) override;
};

class ColorCorrelogramExtractor : public FeatureExtractorInterface {
public:
    Mat extractFeature(const Mat& image) override;
private:
    void quantizeColors(const Mat& src, Mat& dst, int colorLevels = 8);
    void calculateCorrelogram(const Mat& image, std::vector<float>& correlogram, int maxDistance = 5);
};

class SIFTFeatureExtractor : public FeatureExtractorInterface {
public:
    Mat extractFeature(const Mat& image) override;
};

class ORBFeatureExtractor : public FeatureExtractorInterface {
public:
    Mat extractFeature(const Mat& image) override;
};

// Factory class to create feature objects
class FeatureFactory {
public:
    static std::unique_ptr<FeatureExtractorInterface> createFeature(const std::string& featureType) {
        if (featureType == "histogram") {
            return std::make_unique<ColorHistogramExtractor>();
        }
        else if (featureType == "correlogram") {
            return std::make_unique<ColorCorrelogramExtractor>();
        }
        else if (featureType == "sift") {
            return std::make_unique<SIFTFeatureExtractor>();
        }
        else if (featureType == "orb") {
            return std::make_unique<ORBFeatureExtractor>();
        }
        else {
            // Error handling: throw an exception for invalid feature type
            throw std::invalid_argument("Unsupported feature type: " + featureType);
        }
    }
};