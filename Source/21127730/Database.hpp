#pragma once
#include "windows.h "
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <fstream>

using namespace cv;

class FeatureDatabase {
public:
    bool saveFeatures(const std::vector<std::pair<std::string, Mat>>& features, const std::string& featureType, const std::string& dataset, std::string path);
    std::vector<std::pair<std::string, Mat>> loadFeatures(const std::string& featureType, const std::string& dataset, std::string path);
};