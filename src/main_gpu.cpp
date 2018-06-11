//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "mnist/mnist_reader.hpp"
#include <math.h>
#include "main.hpp"

using namespace cv;
using namespace cv::ml;

int main() {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;


    dataset_opencv* data = dataset_to_opencv();

    images_to_features* im_to_feat = get_images_to_features();


    std::vector<std::vector<uint8_t>> features = get_features_images(im_to_feat->training);
    std::vector<std::vector<uint8_t>> features_test = get_features_images(im_to_feat->testing);

    // Create data to classification
    int size_train = 60000;
    int size_test = 10000;
    int size_image = 28 * 28;
    int size_features = features[0].size();

    // Features X
    cv::Mat X_feat_uint = cv::Mat(size_train,size_features,CV_8U);
    cv::Mat X_feat = cv::Mat(size_train,size_features,CV_32F);
    X_feat_uint = vector_to_mat(features);
    X_feat_uint.convertTo(X_feat, CV_32F);
    std::cout << "X_feat: " << X_feat.rows << ": " << X_feat.cols << "\n";

    cv::Mat X_feat_test_uint = cv::Mat(size_test,size_features,CV_8U);
    cv::Mat X_feat_test = cv::Mat(size_test,size_features,CV_32F);
    X_feat_test_uint = vector_to_mat(features_test);
    X_feat_test_uint.convertTo(X_feat_test, CV_32F);
    std::cout << "X_feat_test: " << X_feat_test.rows << ": " << X_feat_test.cols << "\n";


    ////////////////////////
    ////
    ////  Classify MNIST
    ////
    ///////////////////////
    
    
    // Set up RTress for OpenCV 3
    Ptr<RTrees> rtrees = RTrees::create();
    // Set Max Depth
    rtrees->setMaxDepth(10);
    rtrees->setMinSampleCount(2);
    rtrees->setRegressionAccuracy(0);
    rtrees->setUseSurrogates(false);
    rtrees->setMaxCategories(5);
    rtrees->setPriors(cv::Mat());
    rtrees->setCalculateVarImportance(false);
    rtrees->setActiveVarCount(0);
    rtrees->setTermCriteria({ cv::TermCriteria::MAX_ITER, 100, 0 });
     
    // Train RTress on training data 
    Ptr<TrainData> td = TrainData::create(data->X, ROW_SAMPLE, data->Y);
    rtrees->train(td);
     
     
    // Test on a held out test set
    cv::Mat results_float(size_train, 1, CV_32F);
    cv::Mat results(size_train, 1, CV_32S);
    rtrees->predict(data->X_test, results_float);
    results_float.convertTo(results, CV_32S);
    std::cout << "Mnist Trained \n";
    float accuracy = evaluate(results, data->Y_test);
    std::cout << "Accuracy: " << accuracy << "\n";

    ////////////////////////
    ////
    ////  Classify MNIST FEATURES
    ////
    ///////////////////////
    
    
    // Set up RTress for OpenCV 3
    Ptr<RTrees> rtrees_feat = RTrees::create();
    // Set Max Depth
    rtrees_feat->setMaxDepth(10);
    rtrees_feat->setMinSampleCount(2);
    rtrees_feat->setRegressionAccuracy(0);
    rtrees_feat->setUseSurrogates(false);
    rtrees_feat->setMaxCategories(5);
    rtrees_feat->setPriors(cv::Mat());
    rtrees_feat->setCalculateVarImportance(false);
    rtrees_feat->setActiveVarCount(0);
    rtrees_feat->setTermCriteria({ cv::TermCriteria::MAX_ITER, 100, 0 });
     
    // Train RTress on training data 
    Ptr<TrainData> td_feat = TrainData::create(X_feat, ROW_SAMPLE, data->Y);
    rtrees_feat->train(td_feat);
     
     
    // Test on a held out test set
    cv::Mat results_float_feat(size_train, 1, CV_32F);
    cv::Mat results_feat(size_train, 1, CV_32S);
    rtrees->predict(X_feat_test, results_float_feat);
    results_float_feat.convertTo(results_feat, CV_32S);
    std::cout << "Mnist Features Trained \n";
    float accuracy_feat = evaluate(results_feat, data->Y_test);
    std::cout << "Accuracy Featres: " << accuracy_feat << "\n";
    
    return 0;
}

// Util Functions
std::vector<cv::Mat> get_kernels(){
    std::vector<cv::Mat> result;
    int num_kernels = 8;
    int kernel_size = 3;
    double sig = 1, lm = 1, gm = 0.02, ps = 0, theta = 0;
    for (int i = 0; i < num_kernels; i++){
        cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, theta, lm, gm, ps, CV_32F);
        theta += M_PI / (num_kernels);
        result.push_back(kernel);
    }
    return result;
}

std::vector<cv::Mat> apply_filters(std::vector<cv::Mat> kernels, cv::Mat image){
    std::vector<cv::Mat> result;
    for (std::vector<cv::Mat>::iterator it = kernels.begin() ; it != kernels.end(); ++it){
        cv::Mat out = cv::Mat(26,26,CV_32F);
        
        // Create GPU MAT and apply filter
        //cv::cuda::GpuMat gpu_image, gpu_filtered_image;
        //gpu_image.upload(image);
        Ptr<cuda::Convolution> convolver = cuda::createConvolution();
        convolver->convolve(image, *it, out);
        //cv::cuda::filter2D(gpu_image, gpu_filtered_image, gpu_image.depth(), *it);
        //gpu_filtered_image.download(out);

        result.push_back(out);
    }
    return result;
}

std::vector<uint8_t> images_to_vector(std::vector<cv::Mat> images){
    std::vector<uint8_t> result;
    for (std::vector<cv::Mat>::iterator it = images.begin() ; it != images.end(); ++it){
        cv::Mat mat = *it;
        std::vector<uint8_t> vector_current_image;
        vector_current_image.assign(mat.datastart, mat.dataend);
        result.insert(result.end(), vector_current_image.begin(), vector_current_image.end());
    }
    return result;
}

std::vector<cv::Mat> apply_blur(std::vector<cv::Mat> images){
    std::vector<cv::Mat> result;
    for (std::vector<cv::Mat>::iterator it = images.begin() ; it != images.end(); ++it){
        cv::Mat image = *it;
        cv::Mat blur_image = cv::Mat(26,26,CV_32F);
        // Create GPU MAT and apply blur
        cv::cuda::GpuMat gpu_image, gpu_blurred_image;
        gpu_image.upload(image);
        cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(image.type(), image.type(), Size(3, 3), 0.1);
        filter->apply(gpu_image, gpu_blurred_image);
        // cv::cuda::blur(gpu_image, gpu_blurred_image, cv::Size(5,5));
        gpu_blurred_image.download(blur_image);

        result.push_back(blur_image);
    }
    return result;
}


std::vector<std::vector<uint8_t>> get_features_images(std::vector<cv::Mat> images){
    std::vector<std::vector<uint8_t>> result;
    std::vector<cv::Mat> kernels = get_kernels();
    int x = 0;
    for (std::vector<cv::Mat>::iterator it = images.begin() ; it != images.end(); ++it){
        cv::Mat image = *it;
        std::cout << x << "\n";
        std::vector<cv::Mat> filtered_images = apply_filters(kernels, image);
        std::vector<cv::Mat> blurred_filtered_images = apply_blur(filtered_images);
        std::vector<uint8_t> filtered_images_vector = images_to_vector(blurred_filtered_images);
        result.push_back(filtered_images_vector);
        x++;
    }
    return result;
}

float evaluate(cv::Mat& predicted, cv::Mat& actual) {
    assert(predicted.rows == actual.rows);
    int t = 0;
    int f = 0;
    for(int i = 0; i < actual.rows; i++) {
        int p = predicted.at<int>(i,0);
        int a = actual.at<int>(i,0);
        if(a == p) {
            t++;
        } else {
            f++;
        }
    }
    std::cout << "T: " << t << " F: " << f << "\n";
    return (t * 1.0) / (t + f);
}