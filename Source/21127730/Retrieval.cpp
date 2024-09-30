#include "Retrieval.hpp"


bool compareByScore(const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
    return a.second > b.second; // assuming smaller score means more similar
}

// Function to compute cosine similarity between two histograms
double computeCosineSimilarity(const Mat& hist1, const Mat& hist2) {
    CV_Assert(hist1.type() == hist2.type());
    CV_Assert(hist1.size() == hist2.size());

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < hist1.rows; ++i) {
        double bin_i = hist1.at<float>(i);
        double bin_j = hist2.at<float>(i);

        dotProduct += bin_i * bin_j;
        normA += bin_i * bin_i;
        normB += bin_j * bin_j;
    }

    double similarity = dotProduct / (std::sqrt(normA) * std::sqrt(normB));
    return similarity;
}

std::vector<std::string> findTopSimilarImages(const Mat& query_image, FeatureDatabase db, const Mat& queryHistogram, const std::string& featureType, const std::string& dataset, int numResults, std::string& path) {
    std::vector<std::string> topSimilarImages;

    // Load database features
    std::vector<std::pair<std::string, Mat>> databaseFeatures = db.loadFeatures(featureType, dataset, path);

    // Check if features are loaded
    if (databaseFeatures.empty()) {
        std::cerr << "No features loaded from the database." << std::endl;
        return topSimilarImages;
    }

    std::vector<std::pair<std::string, double>> similarityScores;

    // Compute similarity scores 
    for (const auto& dbFeature : databaseFeatures) {
        double score = computeCosineSimilarity(queryHistogram, dbFeature.second);
        similarityScores.push_back({ dbFeature.first, score });
    }

    // Sort similarity scores
    if (!similarityScores.empty()) {
        std::sort(similarityScores.begin(), similarityScores.end(), compareByScore);
    }
    else {
        std::cerr << "No matching features found in the database." << std::endl;
        return topSimilarImages;
    }

    // Retrieve top N results
    for (int i = 0; i < std::min(numResults, static_cast<int>(similarityScores.size())); ++i) {
        topSimilarImages.push_back(similarityScores[i].first);
    }

    return topSimilarImages;
}

std::vector<std::string> retrivalSIFTHistogram(const Mat& queryImage, FeatureDatabase db, const Mat& querySift, const Mat& queryHistogram, const std::string& dataset, int numResults, std::string& path) {
    std::vector<std::string> topSimilarImages;

    // Load database features
    std::vector<std::pair<std::string, Mat>> siftFeatures = db.loadFeatures("sift_histogram", dataset, path);
    std::vector<std::pair<std::string, Mat>> histogramFeatures = db.loadFeatures("histogram", dataset, path);

    // Check if features are loaded
    if (siftFeatures.empty() || histogramFeatures.empty()) {
        std::cerr << "No features loaded from the database." << std::endl;
        return topSimilarImages;
    }

    std::vector<std::pair<std::string, double>> similarityScores;

    // Compute similarity scores for SIFT histograms
    std::map<std::string, double> siftScores;
    for (const auto& dbFeature : siftFeatures) {
        double score = computeCosineSimilarity(querySift, dbFeature.second);
        siftScores[dbFeature.first] = score;
    }

    // Compute similarity scores for color histograms
    std::map<std::string, double> histogramScores;
    for (const auto& dbFeature : histogramFeatures) {
        double score = computeCosineSimilarity(queryHistogram, dbFeature.second);
        histogramScores[dbFeature.first] = score;
    }

    // Combine similarity scores
    for (const auto& siftScore : siftScores) {
        double combinedScore = 0.5 * siftScore.second + 0.5 * histogramScores[siftScore.first]; // equal weighting
        similarityScores.push_back({ siftScore.first, combinedScore });
    }

    // Sort similarity scores
    if (!similarityScores.empty()) {
        std::sort(similarityScores.begin(), similarityScores.end(), compareByScore);
    }
    else {
        std::cerr << "No matching features found in the database." << std::endl;
        return topSimilarImages;
    }

    // Retrieve top N results
    for (int i = 0; i < std::min(numResults, static_cast<int>(similarityScores.size())); ++i) {
        topSimilarImages.push_back(similarityScores[i].first);
    }

    return topSimilarImages;
}


// Function to display query image and retrieved images in separate windows
void displayImagesInSeparateWindows(const std::string& queryImagePath, const std::vector<std::string>& imagePaths) {
    // Create windows to display query image and similar images
    cv::namedWindow("Query Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Similar Images", cv::WINDOW_NORMAL);

    // Load and display query image
    cv::Mat queryImage = cv::imread(queryImagePath, cv::IMREAD_COLOR);
    if (queryImage.empty()) {
        std::cerr << "Failed to read query image: " << queryImagePath << std::endl;
        return;
    }
    cv::imshow("Query Image", queryImage);

    // Customizations (adjust as needed)
    int margin = 5; // Spacing between images (pixels)

    // Calculate grid size for layout (ceil(sqrt(N)) x ceil(sqrt(N)) grid)
    int numImages = imagePaths.size(); // Only retrieved images
    int numCols = std::ceil(std::sqrt(numImages));
    int numRows = std::ceil(static_cast<double>(numImages) / numCols);

    // Calculate the size of the grid cells (maximum width and height)
    int maxWidth = 0;
    int maxHeight = 0;
    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (!image.empty()) {
            maxWidth = std::max(maxWidth, image.cols);
            maxHeight = std::max(maxHeight, image.rows);
        }
    }

    // Include margins in the cell size
    int cellWidth = maxWidth + 2 * margin;
    int cellHeight = maxHeight + 2 * margin;

    // Create a canvas to hold all similar images
    cv::Mat canvas = cv::Mat::zeros(numRows * cellHeight, numCols * cellWidth, queryImage.type());

    // Function to insert an image into the canvas
    auto insertImage = [&](const cv::Mat& image, int row, int col) {
        int x = col * cellWidth + margin;
        int y = row * cellHeight + margin;
        cv::Rect roi(x, y, image.cols, image.rows);
        image.copyTo(canvas(roi));
        };

    // Insert each similar image into the grid
    int index = 0;
    for (const auto& imagePath : imagePaths) {
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << imagePath << std::endl;
            continue;
        }

        int row = index / numCols;
        int col = index % numCols;
        insertImage(image, row, col);

        ++index;
    }

    // Display the grid of similar images in the window
    cv::imshow("Similar Images", canvas);
    cv::waitKey(0);
}