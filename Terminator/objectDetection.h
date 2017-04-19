#pragma once
#include <vector>
#include "objectClass.h"
#include <opencv2\imgproc.hpp>

std::vector<Object> ObjectDetection(cv::Mat img);
cv::Mat OverlayImage(cv::Mat background, cv::Mat foreground, cv::Point location);
void ShowObject(cv::Mat img, std::vector<Object> objects);