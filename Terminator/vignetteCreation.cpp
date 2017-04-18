#include "vignetteCreation.h"
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
CvRect zoom_area;
int interpolation_type = CV_INTER_LINEAR;

cv::Mat GetVignette(cv::Mat img, std::vector<Face> faces)
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
	cv::Mat vignetteImg = img.clone();
	vignetteImg.setTo(cv::Scalar(0, 0, 0));

	if (faces.size() > 0)
		vignetteImg = img(faces[0].face.outterRect);

	cv::Size size(100, 100);//the dst image size,e.g.100x100
	cv::Mat dst;//src image
	resize(vignetteImg, dst, size);//resize image

	dst.copyTo(img(cv::Rect(10, 10, dst.cols, dst.rows)));
	cv::imshow("cam2", img);

	return img;
}