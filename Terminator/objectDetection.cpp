#include "objectDetection.h"
#include "terminator.h"
#include <vector>

#include "opencv2\features2d.hpp"

using namespace cv;
using namespace std;

std::vector<Object> ObjectDetection(cv::Mat img)
{

	return std::vector<Object>(); 

	Mat gray_image, treshold_image, blur;
	img = imread("blob.jpg");

	//cvtColor(img, gray_image, CV_BGR2GRAY);
	cvtColor(img, gray_image, COLOR_RGB2GRAY);
	threshold(gray_image, treshold_image, 55, 255, ThresholdTypes::THRESH_BINARY);
	GaussianBlur(treshold_image, treshold_image, Size(7, 7), 0, 0, BORDER_DEFAULT);
	threshold(treshold_image, treshold_image, 5, 255, ThresholdTypes::THRESH_BINARY);


	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	#pragma region Parameters Blob Detector
	// Change thresholds
	params.minThreshold = 0;
	params.maxThreshold = 20;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 2000;

	// Filter by Circularity
	params.filterByCircularity = false;
	//params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.01;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;
	params.maxInertiaRatio = 1;
	#pragma endregion


	// Storage for blobs
	vector<KeyPoint> keypoints;

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs
	detector->detect(treshold_image, keypoints);

	std::vector<Object> listBlob;


	Mat imgResult, matA;
	Mat rgb[3];
	split(img, rgb);

	Mat rgbaImg[4] = { Mat(img.size(),CV_8UC1,Scalar(0)),Mat(img.size(),CV_8UC1,Scalar(0)),rgb[2],Mat(img.size(),CV_8UC1,Scalar(1)) };
	merge(rgbaImg, 4, imgResult);


	for (int i = 0; i < keypoints.size(); i++)
	{
		Object blob;

		blob.barycentre = point(keypoints[i].pt.x, keypoints[i].pt.y);
		//keypoints[i].size = keypoints[i].size + keypoints[i].size;

		blob.outterRect.x = keypoints[i].pt.x - keypoints[i].size;
		blob.outterRect.y = keypoints[i].pt.y - keypoints[i].size;
		blob.outterRect.width  = keypoints[i].size * 2;
		blob.outterRect.height = keypoints[i].size * 2;
		
		listBlob.push_back(blob);

		Mat subImage(img, cv::Rect(blob.outterRect.x, blob.outterRect.y, blob.outterRect.width, blob.outterRect.height));
		Mat subImageGray(subImage);
		GaussianBlur(subImageGray, subImageGray, Size(3, 3), 0, 0, BORDER_DEFAULT);

		/// Convert it to gray
		cvtColor(subImageGray, subImageGray, COLOR_RGB2GRAY);

		/// Generate grad_x and grad_y
		Mat grad_x, grad_y;
		Mat abs_grad_x, abs_grad_y;

		int scale = 2;
		int delta = 0;
		int ddepth = CV_16S;
		Mat grad;

		/// Gradient X
		Sobel(subImageGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		/// Gradient Y
		Sobel(subImageGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		/// Total Gradient (approximate)
		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		Mat tmp, dst, alpha;

		Mat rgba[4] = { grad,grad,grad,grad };
		merge(rgba, 4, dst);
		imshow("dst", dst);
		imgResult = overlayImage(imgResult, dst, Point(blob.outterRect.x, blob.outterRect.y));
		//dst.copyTo(imgResult(cv::Rect(blob.outterRect.x, blob.outterRect.y, dst.cols, dst.rows)));
	}

	imshow("imgResult", imgResult);
	Mat im_with_keypoints, treshold_image_key;
	drawKeypoints(img, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(treshold_image, keypoints, treshold_image_key, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);
	//imshow("treshold_image", treshold_image_key);

	waitKey();

	return listBlob;
}

Mat overlayImage(Mat background, Mat foreground, Point location)
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