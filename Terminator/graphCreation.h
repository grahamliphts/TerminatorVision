#pragma once
#include <opencv2/imgproc.hpp>

cv::Mat GetGraph(cv::Mat img);
std::vector<cv::Mat> GetGraphSplitChannels(cv::Mat img);

cv::Mat GetGraphTesting(cv::Mat img);