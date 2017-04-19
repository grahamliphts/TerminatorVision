#include "vignetteCreation.h"
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
CvRect zoom_area;
int interpolation_type = CV_INTER_LINEAR;

cv::Mat& GetVignette(cv::Mat& img, std::vector<Face>& faces, cv::Mat& vignette)
{
	float zoom = 5;
	IplImage iplimg = img;
	IplImage *zoomed = cvCloneImage(&iplimg);

	std::vector<IplImage*> vignettes;
	/*if(faces.size() > 0)
		zoom_area = faces[0].face.outterRect;

	if (zoom > 1 && zoom_area.height != 0)
	{
		// calculate a zoom sub-region (in the centre of the image)

		//zoom_area.x = cvFloor((((iplimg.width / zoom) * (zoom / 2.0)) - ((iplimg.width / zoom) / 2.0)));
		//zoom_area.y = cvFloor((((iplimg.height / zoom) * (zoom / 2.0)) - ((iplimg.height / zoom) / 2.0)));
		//
		//zoom_area.width = cvFloor((iplimg.width / zoom));
		//zoom_area.height = cvFloor((iplimg.height / zoom));

		// use ROI settings to zoom into it 

		cvSetImageROI(&iplimg, zoom_area);
		cvResize(&iplimg, zoomed, interpolation_type);
		cvResetImageROI(&iplimg);

	}
	else 
		cvCopy(&iplimg, zoomed, NULL);

	cv::Mat zoomedMat = cv::cvarrToMat(zoomed);*/
	cv::Mat subVignetteImg = img.clone();
	subVignetteImg.setTo(cv::Scalar(0, 0, 0));

	cv::Mat vignetteImg = img.clone();
	vignetteImg.setTo(cv::Scalar(255, 255, 255));

	if (faces.size() > 0)
	{
		vignetteImg = img(faces[0].face.outterRect);
		if (faces[0].leftEye.outterRect.width > 0)
		{
			subVignetteImg = img(faces[0].leftEye.outterRect);
		}
	}


	cv::Size size(150, 150);//the dst image size,e.g.100x100
	cv::Mat vignetteDst;//src image
	resize(vignetteImg, vignetteDst, size);//resize image

	cv::Size subSize(50, 50);//the dst image size,e.g.100x100
	cv::Mat subVignetteDst;//src image
	resize(subVignetteImg, subVignetteDst, subSize);//resize image

	subVignetteDst.copyTo(vignetteDst(cv::Rect(vignetteDst.rows - subSize.width, 0, subVignetteDst.cols, subVignetteDst.rows)));
	vignetteDst.copyTo(img(cv::Rect(10, 10, vignetteDst.cols, vignetteDst.rows)));

	int fontFace = cv::FONT_HERSHEY_DUPLEX;
	int fontScale = 1;
	cv::putText(img, "Name : ", cv::Point(20 + size.width, 40), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);
	cv::putText(img, "Age : ", cv::Point(20 + size.width, 80), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);

	cv::namedWindow("Final", 0);
	cv::resizeWindow("Final", 680, 400);
	cv::imshow("Final", img);

	return vignetteDst;
}