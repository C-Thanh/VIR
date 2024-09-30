#include "Database.hpp"
#include <iostream>
#include <filesystem>

bool FeatureDatabase::saveFeatures(const std::vector<std::pair<std::string, Mat>>& features, const std::string& featureType, const std::string& dataset, std::string path) {
    std::string filename = path + featureType + "_" + dataset + ".xml";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_XML);

    if (!fs.isOpened()) {
        std::cerr << "Failed to open file for writing: " << cv::format("%s", filename.c_str()) << std::endl;
        return false;
    }

    fs << "features" << "[";
    for (const auto& feature_pair : features) {
        const std::string& filename = feature_pair.first;
        const Mat& feature = feature_pair.second;

        // Start a map for each feature
        fs << "{";
        fs << "filename" << filename;
        fs << "feature" << feature;
        // End the map
        fs << "}";
    }
    fs << "]";

    fs.release();
    return true;
}

std::vector<std::pair<std::string, Mat>> FeatureDatabase::loadFeatures(const std::string& featureType, const std::string& dataset, std::string path) {
    std::vector<std::pair<std::string, Mat>> features;

    std::string filename = path + featureType + "_" + dataset + ".xml";
    cv::FileStorage fs(filename, cv::FileStorage::READ | cv::FileStorage::FORMAT_XML);

    if (!fs.isOpened()) {
        std::cerr << "Failed to open file for reading: " << cv::format("%s", filename.c_str()) << std::endl;
        return features;
    }

    cv::FileNode featuresNode = fs["features"];
    if (featuresNode.type() != cv::FileNode::SEQ) {
        std::cerr << "Invalid format in the features file." << std::endl;
        return features;
    }

    for (auto it = featuresNode.begin(); it != featuresNode.end(); ++it) {
        std::string imageFilename;
        Mat feature;

        cv::FileNode node = *it;

        node["filename"] >> imageFilename;
        node["feature"] >> feature;

        features.emplace_back(imageFilename, feature);
    }

    fs.release();
    return features;
}



