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
	gray_image = imread("blob.jpg", IMREAD_GRAYSCALE);

	//cvtColor(img, gray_image, CV_BGR2GRAY);
	threshold(gray_image, treshold_image, 70, 255, ThresholdTypes::THRESH_BINARY);
	GaussianBlur(treshold_image, treshold_image, Size(7, 7), 0, 0, BORDER_DEFAULT);
	threshold(treshold_image, treshold_image, 10, 255, ThresholdTypes::THRESH_BINARY);

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

	Mat im_with_keypoints, treshold_image_key;
	drawKeypoints(gray_image, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(treshold_image, keypoints, treshold_image_key, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);
	imshow("treshold_image", treshold_image_key);


	std::vector<Object> listBlob;
	for (int i = 0; i < keypoints.size(); i++)
	{
		Object blob;

		blob.barycentre = point(keypoints[i].pt.x, keypoints[i].pt.y);

		blob.outterRect.x = keypoints[i].pt.x - keypoints[i].size;
		blob.outterRect.y = keypoints[i].pt.y - keypoints[i].size;
		blob.outterRect.width  = keypoints[i].size;
		blob.outterRect.height = keypoints[i].size;
		
		listBlob.push_back(blob);
	}

	waitKey();

	return listBlob;
}
