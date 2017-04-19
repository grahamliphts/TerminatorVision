#pragma once
#include <vector>
#include <opencv2\imgproc.hpp>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect.hpp>
#include "faceClass.h"
#include "pointClass.h"
#include "classifiersClass.h"
#include <ostream>
#include <iostream>

std::vector<Face> FaceDetection(cv::Mat,Classifiers*);
std::vector<cv::Rect> Getfaces(cv::CascadeClassifier*,cv::Mat,int,int);
std::vector<cv::Rect> GetEyes(cv::CascadeClassifier*, cv::Mat, int, int);
std::vector<cv::Rect> GetMouth(cv::CascadeClassifier*, cv::Mat, int, int);
std::vector<cv::Rect> GetSmile(cv::CascadeClassifier*, cv::Mat img, int minSize, int maxSize);
std::vector<cv::Rect> GetNoze(cv::CascadeClassifier*, cv::Mat img, int minSize, int maxSize);