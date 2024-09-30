#include "Evaluation.hpp"

// Function to trim whitespace from a string
std::string trim_space(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}

// Function to load CSV data into a map of building name to picture names
std::map<std::string, std::set<std::string>> load_csv(const std::string& file_path) {
    std::ifstream file(file_path);
    std::map<std::string, std::set<std::string>> ground_truth;
    std::string line, picture_name, label;

    if (file.is_open()) {
        // Read the header line and find the indices of "Picture Name" and "Building Name"
        std::getline(file, line);
        std::stringstream ss_header(line);
        std::vector<std::string> header;
        std::string column;
        while (std::getline(ss_header, column, ',')) {
            header.push_back(trim_space(column));
        }

        int picture_name_index = 0;
        int label_index = 1;

        // Read the data lines
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<std::string> row;
            std::string cell;
            while (std::getline(ss, cell, ',')) {
                row.push_back(trim_space(cell));
            }

            if (row.size() > std::max(picture_name_index, label_index)) {
                picture_name = row[picture_name_index];
                label = row[label_index];
                ground_truth[label].insert(picture_name);
            }
        }
    }
    return ground_truth;
}

// Function to calculate precision at k
std::vector<double> calculate_precision_at_k(const std::vector<std::string>& retrieved_list, const std::set<std::string>& relevant_set) {
    std::vector<double> precisions;
    int num_relevant = relevant_set.size();
    int correct_retrievals = 0;
    for (size_t k = 0; k < retrieved_list.size(); ++k) {
        if (relevant_set.find(retrieved_list[k]) != relevant_set.end()) {
            correct_retrievals++;
            precisions.push_back(static_cast<double>(correct_retrievals) / num_relevant);
        }
    }
    return precisions;
}

// Function to calculate Average Precision (AP)
double calculate_ap(const std::vector<double>& precisions) {
    if (precisions.empty()) return 0.0;
    double sum = 0.0;
    for (double precision : precisions) {
        sum += precision;
    }
    return sum / precisions.size();
}

std::string get_image_name(const std::string& path) {
    std::string filename = std::filesystem::path(path).stem().string(); // Remove the extension
    filename.erase(0, filename.find_first_not_of('0')); // Remove leading zeros
    return filename;
}

// Function to calculate Mean Average Precision (MAP) for a single query
double calculate_map(const std::string& query_image, const std::vector<std::string>& retrieved_list, const std::map<std::string, std::set<std::string>>& ground_truth) {
    std::string query_image_name = get_image_name(query_image);
    std::string building_name;
    for (const auto& pair : ground_truth) {
        if (pair.second.find(query_image_name) != pair.second.end()) {
            building_name = pair.first;
            break;
        }
    }

    if (building_name.empty()) {
        std::cerr << "Query image not found in ground truth." << std::endl;
        return 0.0;
    }

    std::set<std::string> relevant_set = ground_truth.at(building_name);
    std::vector<double> precisions = calculate_precision_at_k(retrieved_list, relevant_set);
    return calculate_ap(precisions);
}