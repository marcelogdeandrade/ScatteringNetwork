#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mnist/mnist_reader.hpp"
#include <math.h>


struct dataset_opencv {
    cv::Mat X;
    cv::Mat X_test;
    cv::Mat Y;
    cv::Mat Y_test;
};

struct images_to_features {
    std::vector<cv::Mat> training;
    std::vector<cv::Mat> testing;
};
std::vector<cv::Mat> gpu_to_images(std::vector<cv::cuda::GpuMat> images);

std::vector<cv::cuda::GpuMat> apply_filters(std::vector<cv::Mat> kernels, cv::cuda::GpuMat image);

std::vector<uint8_t> images_to_vector(std::vector<cv::Mat> images);

cv::Mat vector_to_mat(std::vector<std::vector<uint8_t>> vec);

std::vector<cv::Mat> apply_blur(std::vector<cv::Mat> images);

std::vector<std::vector<uint8_t>> get_features_images(std::vector<cv::Mat> images);

dataset_opencv* dataset_to_opencv();

images_to_features* get_images_to_features();

std::vector<cv::cuda::GpuMat> images_to_gpu(std::vector<cv::Mat> images);

float evaluate(cv::Mat& predicted, cv::Mat& actual);

#endif