#include "Codebook.hpp"

Mat ClusteringFeature(const std::vector<std::pair<std::string, Mat>>& features, int k) {
    // Extract all feature descriptors into a separate vector
    std::vector<Mat> descriptors;
    for (const auto& feature : features) {
        descriptors.push_back(feature.second);
    }

    // Concatenate all feature descriptors into a single matrix
    Mat allDescriptors;
    for (const auto& desc : descriptors) {
        allDescriptors.push_back(desc);
    }

    // Convert descriptors to CV_32F type
    allDescriptors.convertTo(allDescriptors, CV_32F);

    // K-means clustering
    Mat labels;
    Mat centers;
    kmeans(allDescriptors, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.01), 3, KMEANS_PP_CENTERS, centers);

    return centers;
}


Mat CalculateQueryHistograms(Mat& feature, const Mat& centers) {
    // Initialize histogram
    Mat histogram = Mat::zeros(1, centers.rows, CV_32F);
    centers.convertTo(centers, CV_32F);
    feature.convertTo(feature, CV_32F);

    // Loop through feature rows
    for (int i = 0; i < feature.rows; ++i) {
        Mat feature_row = feature.row(i);

        // Find closest center and update histogram
        double min_dist = DBL_MAX;
        int min_idx = -1;
        for (int j = 0; j < centers.rows; ++j) {
            double dist = cv::norm(feature_row, centers.row(j), cv::NORM_L2);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        histogram.at<float>(0, min_idx)++;
    }
    histogram /= sum(histogram)[0];

    return histogram;
}

std::vector<std::pair<std::string, Mat>> CalculateHistograms(const std::vector<std::pair<std::string, Mat>>& features, const Mat& centers) {
    std::vector<std::pair<std::string, Mat>> histograms;

    // Ensure centers are of type CV_32F
    Mat centersFloat;
    centers.convertTo(centersFloat, CV_32F);

    // Loop through features and calculate histograms
    for (const auto& feature_pair : features) {
        const std::string& image_filename = feature_pair.first;
        const Mat& feature = feature_pair.second;

        // Ensure feature is of type CV_32F
        Mat featureFloat;
        feature.convertTo(featureFloat, CV_32F);

        // Initialize histogram
        Mat histogram = Mat::zeros(1, centersFloat.rows, CV_32F);

        // Loop through feature rows
        for (int i = 0; i < featureFloat.rows; ++i) {
            Mat feature_row = featureFloat.row(i);

            // Find closest center and update histogram
            double min_dist = std::numeric_limits<double>::max();
            int min_idx = -1;
            for (int j = 0; j < centersFloat.rows; ++j) {
                double dist = cv::norm(feature_row, centersFloat.row(j), cv::NORM_L2);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = j;
                }
            }
            histogram.at<float>(0, min_idx)++;
        }
        histogram /= sum(histogram)[0];

        // Add filename-histogram pair
        histograms.emplace_back(image_filename, histogram);
    }

    return histograms;
}




void saveCodebookToFile(const Mat& centers, const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    fs << "features" << centers;
    fs.release();
}

Mat readCodebookFromFile(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return Mat();
    }

    Mat centers;
    fs["features"] >> centers;
    fs.release();

    if (centers.empty()) {
        std::cerr << "Failed to read the codebook from file: " << filename << std::endl;
    }

    return centers;
}