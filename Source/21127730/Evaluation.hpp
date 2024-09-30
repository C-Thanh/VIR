#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>
#include <filesystem>

std::map<std::string, std::set<std::string>> load_csv(const std::string& file_path);
std::string get_image_name(const std::string& path);
double calculate_map(const std::string& query_image, const std::vector<std::string>& retrieved_list, const std::map<std::string, std::set<std::string>>& ground_truth);