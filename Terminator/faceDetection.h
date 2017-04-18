#pragma once
#include <vector>
#include <opencv2\imgproc.hpp>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "faceClass.h"

std::vector<Face> FaceDetection(cv::Mat img);