#pragma once
#include "windows.h "
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

using namespace cv;

Mat ClusteringFeature(const std::vector<std::pair<std::string, Mat>>& features, int k);
std::vector<std::pair<std::string, Mat>> CalculateHistograms(const std::vector<std::pair<std::string, Mat>>& features, const Mat& centers);
Mat CalculateQueryHistograms(Mat& feature, const Mat& centers);
void saveCodebookToFile(const Mat& centers, const std::string& filename);
Mat readCodebookFromFile(const std::string& filename);