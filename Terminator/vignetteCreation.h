#pragma once
#include "objectClass.h"
#include <opencv2\imgproc.hpp>
#include "faceClass.h"

class Vignette
{
public:
	cv::Mat currentFaceVignette;
	cv::Mat currentEyeVignette;

	int size;
	int subSize;

	Vignette(cv::Mat& img);
	cv::Mat Process(cv::Mat& img, std::vector<Face>& faces);
};
