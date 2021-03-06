#pragma once
#include "stdafx.h"
#include <ostream>
#include <iostream>

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

#include "objectDetection.h"
#include "faceDetection.h"
#include "graphCreation.h"
#include "vignetteCreation.h"
#include "classifiersClass.h"

void update(cv::VideoCapture);
void draw(std::vector<Object> , std::vector<Face> , cv::Mat,cv::Mat ,cv::Mat);