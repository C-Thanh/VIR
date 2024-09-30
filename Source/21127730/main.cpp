#include "Processing.hpp"
#include "Codebook.hpp"
#include "Retrieval.hpp"
#include "Evaluation.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <chrono> 

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "<mode> <folderPath/ queryImagePath> <featureType> <dataset>" << std::endl;
    }
    else {
        std::string config_file = "config.ini";
        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> config;
        std::set<std::string> local_features;
        std::set<std::string> global_features;
        std::string database_path, TMBuD_label, CD_label;

        int k = 50;
        int n = 10;
        if (readConfig(config_file, config)) {
            std::stringstream ss(config["FEATURES"]["local"]);
            std::string feature;
            while (getline(ss, feature, ',')) {
                local_features.insert(feature);
            }
            std::stringstream ss2(config["FEATURES"]["global"]);
            while (getline(ss2, feature, ',')) {
                global_features.insert(feature);
            }

            std::cout << "Global features: " << config["FEATURES"]["global"] << std::endl;
            std::cout << "Local features: " << config["FEATURES"]["local"] << std::endl;

            // Extract k and n 
            k = stoi(config["CLUSTER"]["k"]);
            n = stoi(config["RETRIEVE"]["n"]);

            std::cout << "Number of clusters: " << k << std::endl;
            std::cout << "Number of returned images: " << n << std::endl;

            database_path = config["PATH"]["path"] ;
            std::cout << "Path to database: " << database_path << std::endl;

            CD_label = config["CSV"]["CD"];
            std::cout << "Path to CD labels file: " << CD_label << std::endl;
            
            TMBuD_label = config["CSV"]["TMBuD"];
            std::cout << "Path to TMBuD labels file: " << TMBuD_label << std::endl;
        }

        std::string mode = argv[1];

        FeatureDatabase db;
        if (mode == "extract") {
            std::string folderPath = argv[2];
            std::string featureType = argv[3];
            std::string dataset = argv[4];

            if (!checkExist(local_features, featureType) && !checkExist(global_features, featureType)) {
                std::cerr << "Invalid feature type!" << std::endl;
                return 0;
            }

            extractAndSaveFeatures(db, folderPath, featureType, dataset, database_path);

            std::string config_file = "config.ini";  // Replace with your config file path

            // Read local and global features from the config file

            if (checkExist(local_features, featureType)) {
                std::cout << "Creating codebook and plot histogram..." << std::endl;
                clusterAndSaveCodebook(db, featureType, dataset, k, database_path);
                plotAndSaveHistogram(db, featureType, dataset, database_path);
            }

            std::cout << "Finish extracting!" << std::endl;
        }

        else if (mode == "retrieve") {
            std::string queryImagePath = argv[2];
            std::string featureType = argv[3];
            std::string dataset = argv[4];
            std::vector<std::string> topImages;

            if (featureType == "sift_histogram") {
                Mat image = cv::imread(queryImagePath, cv::IMREAD_COLOR);
                if (image.empty()) {
                    std::cerr << "Failed to read image" << std::endl;
                    return 0;
                }
                std::cout << "Read image successful!" << std::endl;

                Mat querySift = extractFeaturesFromImage(image, "sift");
                std::string file_name = database_path + "sift" + "_codebook_" + dataset + ".xml";
                Mat centers = readCodebookFromFile(file_name);
                querySift = CalculateQueryHistograms(querySift, centers);
                Mat queryHistogram = extractFeaturesFromImage(image, "histogram");

                auto start = std::chrono::high_resolution_clock::now();
                topImages = retrivalSIFTHistogram(image, db, querySift, queryHistogram, dataset, n, database_path);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                std::cout << "Total runtime: " << duration.count() << " seconds" << std::endl;

            }
            else {
                if (!checkExist(local_features, featureType) && !checkExist(global_features, featureType)) {
                    std::cerr << "Invalid feature type!" << std::endl;
                    return 0;
                }

                Mat image = cv::imread(queryImagePath, cv::IMREAD_COLOR);
                if (image.empty()) {
                    std::cerr << "Failed to read image" << std::endl;
                    return 0;
                }
                std::cout << "Read image successful!" << std::endl;

                Mat query_feature = extractFeaturesFromImage(image, featureType);

                std::string data = featureType;
                if (checkExist(local_features, featureType)) {
                    std::cout << "Plotting histogram for codebook..." << std::endl;
                    std::string file_name = database_path + featureType + "_codebook_" + dataset + ".xml";
                    Mat centers = readCodebookFromFile(file_name);

                    query_feature = CalculateQueryHistograms(query_feature, centers);

                    data = featureType + "_histogram";
                }
                std::cout << "Extract feature from image successful!" << std::endl;

                auto start = std::chrono::high_resolution_clock::now();
                topImages = findTopSimilarImages(image, db, query_feature, data, dataset, n, database_path);

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                std::cout << "Total runtime: " << duration.count() << " seconds" << std::endl;
            }

            std::vector<std::string> retrieved_filenames;
            for (const auto& path : topImages) {
                retrieved_filenames.push_back(get_image_name(path));
            }

            std::map<std::string, std::set<std::string>> ground_truth;
            if (dataset == "TMBuD") {
                ground_truth = load_csv(TMBuD_label); 
            }
            else if (dataset == "CD") {
                ground_truth = load_csv(CD_label);
            }
            double map_score = calculate_map(queryImagePath, retrieved_filenames, ground_truth);
            std::cout << "MAP score: " << map_score << std::endl;
            
            displayImagesInSeparateWindows(queryImagePath, topImages);
            queryImagePath = "";
        }

        else {
            std::cerr << "Invalid mode" << std::endl;
            return 0;
        }


        return 0;
    }
}
