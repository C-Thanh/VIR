#include "FeatureExtractor.hpp"

//SIFT
Mat SIFTFeatureExtractor::extractFeature(const Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    
    // Convert image to grayscale if it's not already
    Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = image;
    }

    // Initialize SIFT detector
    auto sift = SIFT::create();

    // Detect keypoints and compute descriptors
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(grayImage, cv::noArray(), keypoints, descriptors);

    // Print details about the extracted features
    std::cout << "Extracted features: " << descriptors.rows << " rows, " << descriptors.cols << " cols" << std::endl;

    return descriptors;
}

//ORB
Mat ORBFeatureExtractor::extractFeature(const Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    // Convert image to grayscale if it's not already
    Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = image;
    }

    // Initialize ORB detector
    Ptr<ORB> orb = ORB::create();
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(grayImage, noArray(), keypoints, descriptors);

    return descriptors;
}