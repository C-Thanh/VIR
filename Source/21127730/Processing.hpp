#pragma once
#include "Database.hpp"
#include "Codebook.hpp"
#include "FeatureExtractor.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

bool readConfig(const std::string& filename, std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& config);
bool checkExist(const std::set<std::string> features, std::string feature);
Mat extractFeaturesFromImage(const Mat& image, const std::string& featureType);
void extractAndSaveFeatures(FeatureDatabase db, std::string folderPath, std::string featureType, std::string dataset, std::string path);
void clusterAndSaveCodebook(FeatureDatabase db, std::string featureType, std::string dataset, int k, std::string path);
void plotAndSaveHistogram(FeatureDatabase db, std::string featureType, std::string dataset, std::string path);
