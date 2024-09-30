#include "Processing.hpp"

// Function to trim leading and trailing whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(' ');
    if (first == std::string::npos)
        return ""; // no content

    size_t last = str.find_last_not_of(' ');
    return str.substr(first, last - first + 1);
}

// Function to split a string by a delimiter
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool readConfig(const std::string& filename, std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& config) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening config file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::string currentSection;
    while (getline(file, line)) {
        // Remove leading/trailing whitespace
        line = trim(line);

        // Check for empty line or comment
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Check for section definition
        if (line[0] == '[' && line[line.size() - 1] == ']') {
            // Extract section name
            currentSection = line.substr(1, line.size() - 2);
            config[currentSection] = std::unordered_map<std::string, std::string>();
        }
        else {
            // Split line into key-value pair
            std::vector<std::string> tokens = split(line, '=');
            if (tokens.size() != 2) {
                std::cerr << "Invalid line format in config file: " << line << std::endl;
                continue;
            }
            std::string key = trim(tokens[0]);
            std::string value = trim(tokens[1]);
            config[currentSection][key] = value;
        }
    }
    std::cout << "Read config successfully" << std::endl;

    file.close();
    return true;
}

bool checkExist(const std::set<std::string> features, std::string feature) {
    auto check = features.find(feature);
    if (check == features.end())
        return 0;
    return 1;
}


Mat extractFeaturesFromImage(const Mat& image, const std::string& featureType) {
    std::unique_ptr<FeatureExtractorInterface> featureExtractor = FeatureFactory::createFeature(featureType);

    Mat extractedFeatures;
    try {
        if (!featureExtractor) {
            throw std::runtime_error("Failed to create feature extractor for type: " + featureType);
        }

        extractedFeatures = featureExtractor->extractFeature(image);
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting features: " << e.what() << std::endl;
    }

    return extractedFeatures;
}

void extractAndSaveFeatures(FeatureDatabase db, std::string folderPath, std::string featureType, std::string dataset, std::string path) {
    std::vector<std::pair<std::string, Mat>> allExtractedFeatures;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            std::string imagePath = entry.path().string();
            std::cout << "Processing file: " << imagePath << std::endl;

            Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Failed to read image: " << imagePath << std::endl;
                continue;
            }

            Mat extractedFeatures = extractFeaturesFromImage(image, featureType);
            if (!extractedFeatures.empty()) {
                allExtractedFeatures.push_back(std::make_pair(imagePath, extractedFeatures));
            }
            else {
                std::cerr << "Feature extraction failed for image: " << imagePath << std::endl;
            }
        }
        else {
            std::cerr << "Not a regular file: " << entry.path() << std::endl;
        }
    }

    if (!allExtractedFeatures.empty()) {
        db.saveFeatures(allExtractedFeatures, featureType, dataset, path);
    }

    std::cout << "Features extracted and saved successfully!\n";
}

void clusterAndSaveCodebook(FeatureDatabase db, std::string featureType, std::string dataset, int k, std::string path) {
    std::cout << "Loading features...\n";
    std::vector<std::pair<std::string, Mat>> features = db.loadFeatures(featureType, dataset, path);

    std::cout << "Clustering\n";
    // Clustering features
    Mat centers = ClusteringFeature(features, k);

    std::string file_name = path + featureType + "_codebook_" + dataset + ".xml";

    // Save the codebook to a file
    saveCodebookToFile(centers, file_name);
    std::cout << "Save successfully" << std::endl;
}

void plotAndSaveHistogram(FeatureDatabase db, std::string featureType, std::string dataset, std::string path) {
    std::cout << "Loading features...\n";
    std::vector<std::pair<std::string, Mat>> features = db.loadFeatures(featureType, dataset, path);

    std::string file_name = path + featureType + "_codebook_" + dataset + ".xml";
    Mat centers = readCodebookFromFile(file_name);

    std::cout << "Calculating histogram\n";
    std::vector<std::pair<std::string, Mat>> histograms = CalculateHistograms(features, centers);

    std::string name = featureType + "_histogram";

    if (!histograms.empty()) {
        db.saveFeatures(histograms, name, dataset, path);
    }
    std::cout << "Features extracted and saved successfully!\n";
}
