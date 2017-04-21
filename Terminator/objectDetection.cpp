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
	params.maxThreshold = 80;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 500;
	params.maxArea = 5000;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.01;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
	params.maxInertiaRatio = 1;
	#pragma endregion


	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	/*
	Mat hsvMat, hue;

	cvtColor(img, gray_image, COLOR_RGB2GRAY);
	cvtColor(img, hsvMat, COLOR_RGB2HSV);
	Mat hsv[3];
	split(hsvMat, hsv);
	//GaussianBlur(hsv[0], treshold_black_image, Size(11, 11), 0, 0, BORDER_DEFAULT);

	Canny(gray_image, treshold_black_image, 20, 120, 3);
	//threshold(hsv[0], treshold_black_image, 80, 255, ThresholdTypes::THRESH_BINARY);

	//GaussianBlur(treshold_black_image, treshold_black_image, Size(5, 5), 0, 0, BORDER_DEFAULT);
	cv::imshow("treshold_black_image", treshold_black_image);
	Mat imgResult;
	Mat rgbaImg[4] = { treshold_black_image, treshold_black_image, treshold_black_image, treshold_black_image };
	merge(rgbaImg, 4, imgResult);
	hsv[0] = OverlayImage(hsv[0], imgResult, Point(0, 0));
	cv::imshow("treshold_white_image", hsv[0]);

	for (int j = 0; j < 5; j++)
	{
		params.minThreshold = 51 * j;
		params.maxThreshold = 51 * (j + 1);

		// Set up detector with params
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

		detector->detect(hsv[0], keypoints);

		for (int i = 0; i < keypoints.size(); i++)
		{
			Object blob;

			blob.barycentre = point(keypoints[i].pt.x, keypoints[i].pt.y);

			blob.outterRect.x = keypoints[i].pt.x - keypoints[i].size;
			blob.outterRect.y = keypoints[i].pt.y - keypoints[i].size;
			blob.outterRect.width = keypoints[i].size * 2;
			blob.outterRect.height = keypoints[i].size * 2;

			listBlob.push_back(blob);
		}
		//GaussianBlur(hsv[0], treshold_black_image, Size(11, 11), 0, 0, BORDER_DEFAULT);
	}*/

	
	//cvtColor(img, gray_image, CV_BGR2GRAY);
	cvtColor(img, gray_image, COLOR_RGB2GRAY);
	//Canny(gray_image, treshold_black_image, 50, 100*3, 3);
	threshold(gray_image, treshold_black_image, 75, 255, ThresholdTypes::THRESH_BINARY);
	GaussianBlur(treshold_black_image, treshold_black_image, Size(11, 11), 0, 0, BORDER_DEFAULT);
	detector->detect(treshold_black_image, keypoints);

	for (int i = 0; i < keypoints.size(); i++)
	{
		Object blob;

		blob.barycentre = point(keypoints[i].pt.x, keypoints[i].pt.y);

		blob.outterRect.x = keypoints[i].pt.x - keypoints[i].size;
		blob.outterRect.y = keypoints[i].pt.y - keypoints[i].size;
		blob.outterRect.width = keypoints[i].size * 2;
		blob.outterRect.height = keypoints[i].size * 2;

		listBlob.push_back(blob);
	}

	threshold(gray_image, treshold_white_image, 45, 255, ThresholdTypes::THRESH_BINARY_INV);
	GaussianBlur(treshold_white_image, treshold_white_image, Size(11, 11), 0, 0, BORDER_DEFAULT);

	detector->detect(treshold_white_image, keypoints);

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
	//imshow("treshold_white_image", treshold_white_image);
	//imshow("treshold_black_image", treshold_black_image);

	//Mat im_with_keypoints;
	//drawKeypoints(img, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imshow("keypoints", im_with_keypoints);

	//ShowObject(img, listBlob);
	//waitKey();

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
		if (blob.outterRect.x > 0 && blob.outterRect.y > 0 && blob.outterRect.x + blob.outterRect.width < img.cols && blob.outterRect.y + blob.outterRect.height < img.rows)
		{
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
	}


	imshow("imgResult", imgResult);
}