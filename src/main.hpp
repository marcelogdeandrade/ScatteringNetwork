#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mnist/mnist_reader.hpp"
#include <math.h>

std::vector<cv::Mat> get_kernels();

std::vector<cv::Mat> apply_filters(std::vector<cv::Mat> kernels, cv::Mat image);

std::vector<uint8_t> images_to_vector(std::vector<cv::Mat> images);

cv::Mat vector_to_mat(std::vector<std::vector<uint8_t>> vec);

std::vector<cv::Mat> apply_blur(std::vector<cv::Mat> images);

std::vector<std::vector<uint8_t>> get_features_images(std::vector<cv::Mat> images);

float evaluate(cv::Mat& predicted, cv::Mat& actual);

#endif