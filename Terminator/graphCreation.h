#pragma once
#include <opencv2/imgproc.hpp>

cv::Mat GetGraph(cv::Mat img, int step = 8, int height = 400, int width = 512);
std::vector<cv::Mat> GetGraphSplitChannels(cv::Mat img, int step = 8, int height = 400, int width = 501);

cv::Mat RedPicture(cv::Mat img);

extern std::vector<int> lastObjectCount;
cv::Mat GetGraphObjects(int objCount, int height = 400, int width = 512);

void connectAll(cv::Mat img, int);
