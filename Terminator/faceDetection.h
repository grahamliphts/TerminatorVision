#pragma once
#include <vector>
#include <opencv2\imgproc.hpp>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include "faceClass.h"
#include "pointClass.h"
#include <ostream>
#include <iostream>;

std::vector<Face> FaceDetection(cv::Mat);
std::vector<cv::Rect> Getfaces(std::string,cv::Mat,int,int);
std::vector<cv::Rect> GetEyes(std::string, cv::Mat, int, int);