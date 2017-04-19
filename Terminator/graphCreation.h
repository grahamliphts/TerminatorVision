#pragma once
#include <opencv2/imgproc.hpp>

cv::Mat GetGraph(cv::Mat img, int step = 8);
std::vector<cv::Mat> GetGraphSplitChannels(cv::Mat img, int step = 8);

cv::Mat RedPicture(cv::Mat img);

cv::Mat GetGraphTesting(cv::Mat img, int step = 8);
