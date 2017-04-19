#pragma once
#include "objectClass.h"
#include <opencv2\imgproc.hpp>
#include "faceClass.h"

cv::Mat& GetVignette(cv::Mat& img, std::vector<Face>& faces);