#include "vignetteCreation.h"
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
CvRect zoom_area;
int interpolation_type = CV_INTER_LINEAR;

cv::Mat& GetVignette(cv::Mat& img, std::vector<Face>& faces)
{
	cv::Mat subVignetteImg = img.clone();
	subVignetteImg.setTo(cv::Scalar(0, 0, 0));

	cv::Mat vignetteImg = img.clone();
	vignetteImg.setTo(cv::Scalar(255, 255, 255));

	if (faces.size() > 0)
	{
		vignetteImg = img(faces[0].face.outterRect);
		if (faces[0].leftEye.outterRect.width > 0)
			subVignetteImg = img(faces[0].leftEye.outterRect);
	}

	cv::Size size(150, 150);
	cv::Mat vignetteDst;
	resize(vignetteImg, vignetteDst, size);

	cv::Size subSize(50, 50);
	cv::Mat subVignetteDst;
	resize(subVignetteImg, subVignetteDst, subSize);

	subVignetteDst.copyTo(vignetteDst(cv::Rect(vignetteDst.rows - subSize.width, 0, subVignetteDst.cols, subVignetteDst.rows)));
	vignetteDst.copyTo(img(cv::Rect(10, 10, vignetteDst.cols, vignetteDst.rows)));

	int fontFace = cv::FONT_HERSHEY_DUPLEX;
	int fontScale = 1;
	cv::putText(img, "Name : ", cv::Point(20 + size.width, 40), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);
	cv::putText(img, "Age : ", cv::Point(20 + size.width, 80), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);

	//cv::namedWindow("Final", 0);
	//cv::resizeWindow("Final", 680, 400);
	//cv::imshow("Final", img);

	return vignetteDst;
}