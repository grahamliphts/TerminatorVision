#pragma once
#include <vector>
#include "objectClass.h"
#include <opencv2\imgproc.hpp>

std::vector<Object> ObjectDetection(cv::Mat img);
cv::Mat overlayImage(cv::Mat background, cv::Mat foreground, cv::Point location);