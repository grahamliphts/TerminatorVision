#include "faceDetection.h"

std::vector<Face> FaceDetection(cv::Mat img,Classifiers* haarManager)
{

	int MinFaceSize = 120;
	int eyeArea = 4 ;
	int mouthArea = 2;
	// LOCATING FACES IN IMAGE
	std::vector<Face> faceList;
	std::vector<cv::Rect> foundFaces;
	cv::Point2f center;
	Face tempFace;
	//img = cv::imread("john-cena.jpg", CV_LOAD_IMAGE_COLOR);

	foundFaces = Getfaces(&haarManager->faceClassifier,img, MinFaceSize,400);
	if (foundFaces.size() > 0) {	 
		for (int i = 0; i < foundFaces.size(); i++) {
			center = (foundFaces[i].br() + foundFaces[i].tl())*0.5;
			tempFace.face.outterRect = foundFaces[i];
			tempFace.face.barycentre = point::point(center.x, center.y);
			faceList.push_back(tempFace);
		}
	}

	else
	{
		
		foundFaces = Getfaces(&haarManager->faceSideClassifier, img, MinFaceSize, 400);
		if (foundFaces.size() > 0) {
			for (int i = 0; i < foundFaces.size(); i++) {
				center = (foundFaces[i].br() + foundFaces[i].tl())*0.5;
				tempFace.face.outterRect = foundFaces[i];
				tempFace.face.barycentre = point::point(center.x, center.y);
				faceList.push_back(tempFace);
			}
		}
	}
	
	
	// LOCATING FACE SUBPART IN FACES
	Face curFace;
	cv::Mat FaceImg;
	cv::Mat topFaceImage;
	cv::Rect tempRect;
	cv::Mat bottomFaceImg;
	std::vector<cv::Rect> foundEyes;	
	std::vector<cv::Rect> FoundMouth;
	std::vector<cv::Rect> FoundSmile;
	std::vector<cv::Rect> FoundNoze;

	for(int it = 0; it < faceList.size(); it++)
	{
		foundEyes.clear();
		foundEyes.reserve(2);

		curFace = faceList[it];
		FaceImg = img(curFace.face.outterRect);
		topFaceImage = FaceImg(cv::Rect(0, 0, FaceImg.cols, FaceImg.rows / 2));		
		foundEyes = GetEyes(&haarManager->eyeClassifierGlasses, topFaceImage, MinFaceSize / eyeArea, topFaceImage.rows);

		if (foundEyes.size() > 0) {
			for (int i = 0; i <= foundEyes.size() - 1; i++)
			{
				tempRect = foundEyes[i];
				tempRect.x = curFace.face.outterRect.x + foundEyes[i].x;
				tempRect.y = curFace.face.outterRect.y + foundEyes[i].y;
				if(i == 0)
					faceList[it].leftEye.outterRect = tempRect;
				if(i == 1)
					faceList[it].rightEye.outterRect = tempRect;
				
			}
		}
		else
		{
			foundEyes.clear();
			if (foundEyes.size() > 0 ) {
				foundEyes = GetEyes(&haarManager->eyeClassifierDefault, topFaceImage, MinFaceSize / eyeArea, topFaceImage.rows);
				for (int i = 0; i <= foundEyes.size() - 1; i++)
				{
					tempRect = foundEyes[i];
					tempRect.x = curFace.face.outterRect.x + foundEyes[i].x;
					tempRect.y = curFace.face.outterRect.y + foundEyes[i].y;

					if (i == 0)
					{
						faceList[it].leftEye.outterRect = tempRect;
					}
					if (i == 1)
						faceList[it].rightEye.outterRect = tempRect;
						
				}
			}
		}
		/*
		FoundNoze.clear();
		FoundNoze.reserve(1);

		FoundNoze = GetNoze(&haarManager->nozeClassifier, FaceImg, MinFaceSize / mouthArea, FaceImg.rows / 2);
		if (FoundNoze.size() > 0) {
			tempRect = FoundNoze[0];
			tempRect.x = curFace.face.outterRect.x + FoundNoze[0].x;
			tempRect.y = curFace.face.outterRect.y + FoundNoze[0].y;
			faceList[it].noze.outterRect = tempRect;
		}
		*/

		FoundMouth.clear();
		FoundMouth.reserve(1);
		bottomFaceImg = FaceImg(cv::Rect(0, FaceImg.rows/2, FaceImg.cols, FaceImg.rows / 2));

		FoundMouth = GetMouth(&haarManager->mouthClassifier, bottomFaceImg, MinFaceSize / mouthArea, FaceImg.rows/2);
		if (FoundMouth.size() > 0) {
				tempRect = FoundMouth[0];
				tempRect.x = curFace.face.outterRect.x + FoundMouth[0].x;
				tempRect.y = curFace.face.outterRect.y + FoundMouth[0].y + FaceImg.rows / 2;
				faceList[it].mouse.outterRect = tempRect;
				
			FoundSmile.clear();
			FoundSmile.reserve(1);
			FoundSmile = GetSmile(&haarManager->smileClassifer, bottomFaceImg, MinFaceSize / mouthArea, tempRect.width);
			if (FoundSmile.size() > 0) {
					faceList[it].isSmile = true;
			}
			else
				faceList[it].isSmile = false;
				
		}
		

	}
	

	//Show the results
	for each (Face curFace in faceList)
	{
		rectangle(img, curFace.face.outterRect.br(), curFace.face.outterRect.tl(), cv::Scalar(0, 0, 0), 1, 8, 0); // Draw Face outter rect
		rectangle(img, curFace.leftEye.outterRect.br(), curFace.leftEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Left eye
		rectangle(img, curFace.rightEye.outterRect.br(), curFace.rightEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Right eye
		if(curFace.isSmile)
			rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 255, 255), 1, 8, 0); // Draw mouse
		else
			rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 0, 255), 1, 8, 0); // Draw mouse
		rectangle(img, curFace.noze.outterRect.br(), curFace.noze.outterRect.tl(), cv::Scalar(255, 0, 255), 1, 8, 0); // Draw mouse

	}


	imshow("FaceDetected", img);
	int key2 = cv::waitKey(20);

	return faceList;
}

std::vector<cv::Rect> Getfaces(cv::CascadeClassifier* detector,cv::Mat img,int minSize, int maxSize)
{
	int groundThreshold = 3;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	detector->detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);
	return found;
}

std::vector<cv::Rect> GetEyes(cv::CascadeClassifier* detector, cv::Mat img, int minSize, int maxSize)
{
	int groundThreshold = 2;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	detector->detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);
	return found;
}
std::vector<cv::Rect> GetMouth(cv::CascadeClassifier* detector, cv::Mat img, int minSize, int maxSize)
{
	int groundThreshold = 2;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	detector->detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);
	return found;
}

std::vector<cv::Rect> GetSmile(cv::CascadeClassifier* detector, cv::Mat img, int minSize, int maxSize)
{
	int groundThreshold = 2;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	detector->detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);
	return found;
}

std::vector<cv::Rect> GetNoze(cv::CascadeClassifier* detector, cv::Mat img, int minSize, int maxSize)
{
	int groundThreshold = 2;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	detector->detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | cv::CASCADE_SCALE_IMAGE, minimalObjectSize, maximalObjectSize);
	return found;
}

