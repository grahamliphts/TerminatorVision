#include "vignetteCreation.h"

CvRect zoom_area;
int interpolation_type = CV_INTER_LINEAR;

cv::Mat GetVignette(cv::Mat img)
{
	float zoom = 5;
	IplImage iplimg = img;
	IplImage *zoomed = cvCloneImage(&iplimg);

	std::vector<IplImage*> vignettes;

	if (zoom > 1)
	{
		// calculate a zoom sub-region (in the centre of the image)

		zoom_area.x = cvFloor((((iplimg.width / zoom) * (zoom / 2.0)) - ((iplimg.width / zoom) / 2.0)));
		zoom_area.y = cvFloor((((iplimg.height / zoom) * (zoom / 2.0)) - ((iplimg.height / zoom) / 2.0)));

		zoom_area.width = cvFloor((iplimg.width / zoom));
		zoom_area.height = cvFloor((iplimg.height / zoom));

		// use ROI settings to zoom into it 

		cvSetImageROI(&iplimg, zoom_area);
		cvResize(&iplimg, zoomed, interpolation_type);
		cvResetImageROI(&iplimg);

	}
	else 
		cvCopy(&iplimg, zoomed, NULL);

	cv::Mat zoomedMat = cv::cvarrToMat(zoomed);

	cv::Size size(100, 100);//the dst image size,e.g.100x100
	cv::Mat dst;//src image
	resize(zoomedMat, dst, size);//resize image

	dst.copyTo(img(cv::Rect(10, 10, dst.cols, dst.rows)));

	//vignettes.push_back(zoomed);
	return img;
}