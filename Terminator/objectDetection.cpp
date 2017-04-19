#include "objectDetection.h"
#include "terminator.h"
#include <vector>

#include "opencv2\features2d.hpp"

using namespace cv;
using namespace std;

std::vector<Object> ObjectDetection(cv::Mat img)
{

	//return std::vector<Object>();
	//img = imread("blob.jpg");

	Mat gray_image, treshold_black_image, treshold_white_image, treshold_image;
	vector<KeyPoint> keypoints;
	SimpleBlobDetector::Params params;
	std::vector<Object> listBlob;

	#pragma region Parameters Blob Detector
	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 50;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 3000;

	// Filter by Circularity
	params.filterByCircularity = false;
	//params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.01;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
	params.maxInertiaRatio = 1;
	#pragma endregion

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);


	//cvtColor(img, gray_image, CV_BGR2GRAY);
	cvtColor(img, gray_image, COLOR_RGB2GRAY);
	threshold(gray_image, treshold_black_image, 65, 255, ThresholdTypes::THRESH_BINARY);
	GaussianBlur(treshold_black_image, treshold_black_image, Size(9, 9), 0, 0, BORDER_DEFAULT);
	threshold(treshold_black_image, treshold_black_image, 20, 255, ThresholdTypes::THRESH_BINARY);

	threshold(gray_image, treshold_white_image, 180, 255, ThresholdTypes::THRESH_BINARY_INV);
	GaussianBlur(treshold_white_image, treshold_white_image, Size(9, 9), 0, 0, BORDER_DEFAULT);
	threshold(treshold_white_image, treshold_white_image, 10, 255, ThresholdTypes::THRESH_BINARY);

	bitwise_xor(treshold_white_image, treshold_black_image, treshold_image);
	threshold(treshold_image, treshold_image, 100, 255, ThresholdTypes::THRESH_BINARY_INV);

	detector->detect(treshold_image, keypoints);

	for (int i = 0; i < keypoints.size(); i++)
	{
		Object blob;

		blob.barycentre = point(keypoints[i].pt.x, keypoints[i].pt.y);

		blob.outterRect.x = keypoints[i].pt.x - keypoints[i].size;
		blob.outterRect.y = keypoints[i].pt.y - keypoints[i].size;
		blob.outterRect.width  = keypoints[i].size * 2;
		blob.outterRect.height = keypoints[i].size * 2;
		
		listBlob.push_back(blob);
	}

	imshow("treshold_image", treshold_image);

	Mat im_with_keypoints;
	drawKeypoints(img, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("keypoints", im_with_keypoints);

	ShowObject(img, listBlob);
	waitKey();

	return listBlob;
}

Mat OverlayImage(Mat background, Mat foreground, Point location)
{
	Mat output;
	background.copyTo(output);

	for (int y = max(location.y, 0); y < background.rows; ++y)
	{
		int fY = y - location.y;

		if (fY >= foreground.rows)
			break;

		for (int x = max(location.x, 0); x < background.cols; ++x)
		{
			int fX = x - location.x; 

			if (fX >= foreground.cols)
				break;

			double opacity = ((double)foreground.data[fY * foreground.step + fX * foreground.channels() + 3]) / 255.;

			for (int c = 0; opacity > 0 && c < output.channels(); ++c)
			{
				unsigned char foregroundPx = foreground.data[fY * foreground.step + fX * foreground.channels() + c];
				unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
				output.data[y*output.step + output.channels()*x + c] = backgroundPx * (1. - opacity) + foregroundPx * opacity;
			}
		}
	}

	return output;
}

void ShowObject(cv::Mat img, std::vector<Object> objects)
{
	Mat imgResult;
	Mat rgb[3];
	split(img, rgb);

	Mat rgbaImg[4] = { Mat(img.size(),CV_8UC1,Scalar(0)),Mat(img.size(),CV_8UC1,Scalar(0)),rgb[2],Mat(img.size(),CV_8UC1,Scalar(1)) };
	merge(rgbaImg, 4, imgResult);

	for (int i = 0; i < objects.size(); i++)
	{
		Object blob = objects[i];

		Mat subImage(img, cv::Rect(blob.outterRect.x, blob.outterRect.y, blob.outterRect.width, blob.outterRect.height));
		Mat subImageGray(subImage);
		GaussianBlur(subImageGray, subImageGray, Size(3, 3), 0, 0, BORDER_DEFAULT);

		/// Convert it to gray
		cvtColor(subImageGray, subImageGray, COLOR_RGB2GRAY);

		/// Generate grad_x and grad_y
		Mat grad;
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		int scale = 2;
		int delta = 0;
		int ddepth = CV_16S;

		Sobel(subImageGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		Sobel(subImageGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		Mat dst;
		Mat rgba[4] = { grad,grad,grad,grad };
		merge(rgba, 4, dst);

		imgResult = OverlayImage(imgResult, dst, Point(blob.outterRect.x, blob.outterRect.y));
		rectangle(imgResult, blob.outterRect, Scalar(255, 255, 255));
	}

	imshow("imgResult", imgResult);
}