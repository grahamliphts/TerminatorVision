#pragma once
#include "objectClass.h"
#include <opencv2\imgproc.hpp>
#include "faceClass.h"

class Vignette
{
private:
	double valCompare = 0.3;
public:
	cv::Mat currentFaceVignette;
	cv::Mat currentEyeVignette;

	int size;
	int subSize;

	double sumPixels;
	bool changedFace;

	Vignette(cv::Mat& img);
	cv::Mat Process(cv::Mat& img, std::vector<Face>& faces);
};
