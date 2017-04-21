#include "vignetteCreation.h"
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

Vignette::Vignette(cv::Mat& img)
{
	size = 150;
	subSize = 50;

	changedFace = false;

	currentFaceVignette = img.clone();
	currentFaceVignette.setTo(cv::Scalar(0, 0, 0));

	currentEyeVignette = img.clone();
	currentEyeVignette.setTo(cv::Scalar(255, 255, 255));

	resize(currentFaceVignette, currentFaceVignette, cv::Size(size, size));
	resize(currentEyeVignette, currentEyeVignette, cv::Size(subSize, subSize));

	currentEyeVignette.copyTo(currentFaceVignette(cv::Rect(currentFaceVignette.rows - subSize, 0, currentEyeVignette.cols, currentEyeVignette.rows)));
	currentFaceVignette.copyTo(img(cv::Rect(10, 10, currentFaceVignette.cols, currentFaceVignette.rows)));
}

cv::Mat Vignette::Process(cv::Mat& img, std::vector<Face>& faces)
{
	bool detectFace = false;
	if (faces.size() > 0)
		detectFace = true;

	if (detectFace)
	{
		currentFaceVignette = img.clone();
		currentFaceVignette.setTo(cv::Scalar(0, 0, 0));

		currentFaceVignette = img(faces[0].face.outterRect);
		resize(currentFaceVignette, currentFaceVignette, cv::Size(size, size));

		if (faces[0].leftEye.outterRect.width > 0)
		{
			currentEyeVignette = img.clone();
			currentEyeVignette.setTo(cv::Scalar(255, 255, 255));

			currentEyeVignette = img(faces[0].leftEye.outterRect);
			resize(currentEyeVignette, currentEyeVignette, cv::Size(subSize, subSize));
			currentEyeVignette.copyTo(currentFaceVignette(cv::Rect(currentFaceVignette.rows - subSize, 0, currentEyeVignette.cols, currentEyeVignette.rows)));
		}
		currentEyeVignette.copyTo(currentFaceVignette(cv::Rect(currentFaceVignette.rows - subSize, 0, currentEyeVignette.cols, currentEyeVignette.rows)));
		currentFaceVignette.copyTo(img(cv::Rect(10, 10, currentFaceVignette.cols, currentFaceVignette.rows)));
	}
	else
	{
		currentEyeVignette.copyTo(currentFaceVignette(cv::Rect(currentFaceVignette.rows - subSize, 0, currentEyeVignette.cols, currentEyeVignette.rows)));
		currentFaceVignette.copyTo(img(cv::Rect(10, 10, currentFaceVignette.cols, currentFaceVignette.rows)));
	}

	double s = cv::sum(currentFaceVignette)[0] + cv::sum(currentFaceVignette)[1] + cv::sum(currentFaceVignette)[2];
	double epsilon = sumPixels * valCompare;

	if (fabs(sumPixels - s) < epsilon)
		sumPixels = s;
	else
	{
		sumPixels = s;
		changedFace = true;
	}
	
	return cv::Mat();
}