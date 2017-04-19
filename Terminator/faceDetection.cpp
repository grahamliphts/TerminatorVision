#include "faceDetection.h"

std::vector<Face> FaceDetection(cv::Mat img, Classifiers* haarManager)
{

	int MinFaceSize = 120;
	int eyeArea = 4;
	int mouthArea = 3;
	
	// LOCATING FACES IN IMAGE
	std::vector<Face> faceList;
	std::vector<cv::Rect> foundFacesFront;
	std::vector<cv::Rect> foundFacesSide;
	cv::Point2f center;
	Face tempFace;
	bool FaceShared = false;
	//img = cv::imread("john-cena.jpg", CV_LOAD_IMAGE_COLOR);

	foundFacesFront = Getfaces(&haarManager->faceClassifier, img, MinFaceSize, img.cols);
	if (foundFacesFront.size() > 0) {
		for (int i = 0; i < foundFacesFront.size(); i++) {
			FaceShared = false;
			for(int j=0; j < i; j++)
			{
				FaceShared = (((foundFacesFront[i] & foundFacesFront[j]).area()) > 0);
			}
			if (!FaceShared)
			{
				center = (foundFacesFront[i].br() + foundFacesFront[i].tl())*0.5;
				tempFace.face.outterRect = foundFacesFront[i];
				tempFace.face.barycentre = point::point(center.x, center.y);
				faceList.push_back(tempFace);
			}
		}
	}

	foundFacesSide = Getfaces(&haarManager->faceSideClassifier, img, MinFaceSize, img.cols);
	if (foundFacesSide.size() > 0) {
		for (int i = 0; i < foundFacesSide.size(); i++) {
			FaceShared = false;
			for each (cv::Rect frontface in foundFacesFront)
			{
				FaceShared = (((foundFacesSide[i] & frontface).area()) > 0);
			}
			for (int j = 0; j < i; j++)
			{
				FaceShared = (((foundFacesSide[i] & foundFacesSide[j]).area()) > 0);
			}
			if (!FaceShared)
			{
				center = (foundFacesSide[i].br() + foundFacesSide[i].tl())*0.5;
				tempFace.face.outterRect = foundFacesSide[i];
				tempFace.face.barycentre = point::point(center.x, center.y);
				faceList.push_back(tempFace);
				FaceShared = false;
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

	for (int it = 0; it < faceList.size(); it++)
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
				if (i == 0)
					faceList[it].leftEye.outterRect = tempRect;
				if (i == 1)
					faceList[it].rightEye.outterRect = tempRect;

			}
		}
		else
		{
			foundEyes.clear();
			if (foundEyes.size() > 0) {
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

		FoundNoze.clear();
		FoundNoze.reserve(1);
		int offset = 0;
		FoundNoze = GetNoze(&haarManager->nozeClassifier, FaceImg, MinFaceSize / mouthArea, FaceImg.rows / 2);
		if (FoundNoze.size() > 0) {
			tempRect = FoundNoze[0];
			tempRect.x = curFace.face.outterRect.x + FoundNoze[0].x;
			tempRect.y = curFace.face.outterRect.y + FoundNoze[0].y;
			faceList[it].noze.outterRect = tempRect;

			offset = FoundNoze[0].y + (FoundNoze[0].height * 0.7);
		}


		FoundMouth.clear();
		FoundMouth.reserve(1);
		//sharedNozeMouth = ((FoundMouth[0] & faceList[it].noze.outterRect).area()) > 0;

		if (offset < FaceImg.rows)
		{
			bottomFaceImg = FaceImg(cv::Rect(0, offset, FaceImg.cols, FaceImg.rows - offset));
		}
		else
			bottomFaceImg = FaceImg(cv::Rect(0, FaceImg.rows / 2, FaceImg.cols, FaceImg.rows / 2));

		FoundMouth = GetMouth(&haarManager->mouthClassifier, bottomFaceImg, MinFaceSize / mouthArea, FaceImg.rows / 2);
		if (FoundMouth.size() > 0) {

			tempRect = FoundMouth[0];
			tempRect.x = curFace.face.outterRect.x + FoundMouth[0].x;
			tempRect.y = curFace.face.outterRect.y + FoundMouth[0].y + (offset);
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



	return faceList;
}

std::vector<cv::Rect> Getfaces(cv::CascadeClassifier* detector, cv::Mat img, int minSize, int maxSize)
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

