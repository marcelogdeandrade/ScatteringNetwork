#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "mnist/mnist_reader.hpp"
#include <math.h>
#include "main.hpp"


dataset_opencv* dataset_to_opencv(){
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    dataset_opencv* result = new dataset_opencv();
        // Create data to classification
    int size_train = 60000;
    int size_test = 10000;
    int size_image = 28 * 28;

    cv::Mat X_uint = cv::Mat(size_train,size_image,CV_8U);
    cv::Mat X = cv::Mat(size_train,size_image,CV_32F);
    X_uint = vector_to_mat(dataset.training_images);
    X_uint.convertTo(X, CV_32F);
    result->X = X;
    std::cout << "X: " << result->X.rows << ": " << result->X.cols << "\n";

    cv::Mat X_test_uint = cv::Mat(size_test,size_image,CV_8U);
    cv::Mat X_test = cv::Mat(size_test,size_image,CV_32F);
    X_test_uint = vector_to_mat(dataset.test_images);
    X_test_uint.convertTo(X_test, CV_32F);
    result->X_test = X_test;
    std::cout << "X_test: " << result->X_test.rows << ": " << result->X_test.cols << "\n";

    // Labels
    cv::Mat Y_uint = cv::Mat(size_train, 1, CV_8U);
    cv::Mat Y = cv::Mat(size_train, 1, CV_32S);
    std::memcpy(Y_uint.data, dataset.training_labels.data(), dataset.training_labels.size());
    Y_uint.convertTo(Y, CV_32S);
    result->Y = Y;
    std::cout << "Y " << result->Y.rows << ": " << result->Y.cols << "\n";
		
    cv::Mat Y_test_uint = cv::Mat(size_test, 1, CV_8U);
    cv::Mat Y_test = cv::Mat(size_test, 1, CV_32S);
    std::memcpy(Y_test_uint.data, dataset.test_labels.data(), dataset.test_labels.size());
    Y_test_uint.convertTo(Y_test, CV_32S);
    result->Y_test = Y_test;
    std::cout << "Y_test " << result->Y_test.rows << ": " << result->Y_test.cols << "\n";

    return result;
}

cv::Mat vector_to_mat(std::vector<std::vector<uint8_t>> vec){
    cv::Mat mat(vec.size(), vec.at(0).size(), CV_8U);
    for(int i=0; i<mat.rows; ++i)
        for(int j=0; j<mat.cols; ++j)
            mat.at<uint8_t>(i, j) = vec.at(i).at(j);
    return mat;
}

images_to_features* get_images_to_features(){
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);


    images_to_features* result = new images_to_features();
    std::vector<cv::Mat> training_images_mat;
    std::vector<std::vector<uint8_t>>::iterator it_train;
    for(it_train = dataset.training_images.begin(); it_train != dataset.training_images.end(); it_train++){
        cv::Mat M = cv::Mat(28,28,CV_32F);
        std::vector<uint8_t> aux_vec = *it_train;
        std::memcpy(M.data, aux_vec.data(), aux_vec.size());
        training_images_mat.push_back(M);
    }
    result->training = training_images_mat;

    std::vector<cv::Mat> test_images_mat;
    std::vector<std::vector<uint8_t>>::iterator it_test;
    for(it_test = dataset.test_images.begin(); it_test != dataset.test_images.end(); it_test++){
        cv::Mat M = cv::Mat(28,28,CV_32F);
        std::vector<uint8_t> aux_vec = *it_test;
        std::memcpy(M.data, aux_vec.data(), aux_vec.size());
        test_images_mat.push_back(M);
    }
    result->testing = test_images_mat;
    return result;
}