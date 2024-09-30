#pragma once
#include "Database.hpp"
#include "Codebook.hpp"
#include "Processing.hpp"
#include "FeatureExtractor.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>

std::vector<std::string> findTopSimilarImages(const Mat& query_image, FeatureDatabase db, const Mat& queryHistogram, const std::string& featureType, const std::string& dataset, int numResults, std::string& path);
std::vector<std::string> retrivalSIFTHistogram(const Mat& query_image, FeatureDatabase db, const Mat& query_sift, const Mat& query_histogram, const std::string& dataset, int numResults, std::string& path);
void displayImagesInSeparateWindows(const std::string& queryImagePath, const std::vector<std::string>& imagePaths);