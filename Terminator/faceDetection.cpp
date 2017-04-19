#include "faceDetection.h"

std::vector<Face> FaceDetection(cv::Mat img)
{
	// LOCATING FACES IN IMAGE
	std::vector<Face> faceList;
	std::vector<cv::Rect> foundFaces;
	foundFaces = Getfaces("HaarCascade/haarcascade_frontalface_default.xml",img,200,700);
	if (foundFaces.size() > 0) {
		Face tempFace;
		 
		for (int i = 0; i <= foundFaces.size() - 1; i++) {
			//rectangle(img, foundFaces[i].br(), foundFaces[i].tl(), cv::Scalar(0, 0, 0), 1, 8, 0);
			cv::Point2f center = (foundFaces[i].br() + foundFaces[i].tl())*0.5;
			tempFace.face.outterRect = foundFaces[i];
			tempFace.face.barycentre = point::point(center.x, center.y);
			faceList.push_back(tempFace);
		}
	}

	else
	{
		
		foundFaces = Getfaces("HaarCascade/haarcascade_profileface.xml", img, 200, 700);
		if (foundFaces.size() > 0) {
			Face tempFace;
			for (int i = 0; i <= foundFaces.size() - 1; i++) {
				//rectangle(img, foundFaces[i].br(), foundFaces[i].tl(), cv::Scalar(0, 0, 0), 1, 8, 0);
				cv::Point2f center = (foundFaces[i].br() + foundFaces[i].tl())*0.5;
				tempFace.face.outterRect = foundFaces[i];
				tempFace.face.barycentre = point::point(center.x, center.y);
				faceList.push_back(tempFace);
			}
		}
	}
	
	
	// LOCATING FACE SUBPART IN FACES
	for(int it = 0; it < faceList.size(); it++)
	{
		Face curFace = faceList[it];
		cv::Mat FaceImg;
		FaceImg = img(curFace.face.outterRect);
		//cv::imshow("face", FaceImg);

		std::vector<cv::Rect> foundEyes;
		foundEyes = GetEyes("HaarCascade/haarcascade_eye_tree_eyeglasses.xml", FaceImg, 50, 100);
		if (foundEyes.size() > 0) {
			for (int i = 0; i <= foundEyes.size() - 1; i++)
			{
				//rectangle(FaceImg, foundEyes[i].br(), foundEyes[i].tl(), cv::Scalar(0, 255, 0), 1, 8, 0);
				cv::Rect tempRect = foundEyes[i];
				tempRect.x = curFace.face.outterRect.x + foundEyes[i].x;
				tempRect.y = curFace.face.outterRect.y + foundEyes[i].y;

				if(i == 0)
					faceList[i].leftEye.outterRect = tempRect;
				if(i == 1)
					faceList[i].rightEye.outterRect = tempRect;
			}
		}
		else
		{
			foundEyes.clear();
			if (foundEyes.size() > 0 ) {
				foundEyes = GetEyes("HaarCascade/haarcascade_eye.xml", FaceImg, 50, 100);
				for (int i = 0; i <= foundEyes.size() - 1; i++)
				{
					//rectangle(FaceImg, foundEyes[i].br(), foundEyes[i].tl(), cv::Scalar(0, 255, 0), 1, 8, 0);
					cv::Rect tempRect = foundEyes[i];
					tempRect.x = curFace.face.outterRect.x + foundEyes[i].x;
					tempRect.y = curFace.face.outterRect.y + foundEyes[i].y;

					if (i == 0)
					{
						faceList[i].leftEye.outterRect = tempRect;
					}
					if (i == 1)
						faceList[i].rightEye.outterRect = tempRect;
				}
			}
		}

	}
	

	//Show the results
	//std::cout << faceList.size() <<std::endl;
	for each (Face curFace in faceList)
	{
		rectangle(img, curFace.face.outterRect.br(), curFace.face.outterRect.tl(), cv::Scalar(0, 0, 0), 1, 8, 0); // Draw Face outter rect
		rectangle(img, curFace.leftEye.outterRect.br(), curFace.leftEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Left eye
		rectangle(img, curFace.rightEye.outterRect.br(), curFace.rightEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Right eye
		rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 0, 255), 1, 8, 0); // Draw mouse
		rectangle(img, curFace.noze.outterRect.br(), curFace.noze.outterRect.tl(), cv::Scalar(255, 0, 0), 1, 8, 0); // Draw mouse

	}


	imshow("FaceDetected", img);
	int key2 = cv::waitKey(20);

	return faceList;
}

std::vector<cv::Rect> Getfaces(std::string HaarCascade,cv::Mat img,int minSize, int maxSize)
{
	cv::CascadeClassifier detector;
	std::string cascadeName = HaarCascade;;
	bool loaded = detector.load(cascadeName);
	// Parameters of detectMultiscale Cascade Classifier
	int groundThreshold = 1;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	// Detect faces
	detector.detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | 2, minimalObjectSize, maximalObjectSize);
	// Draw the results into mat retrieved from webcam
	return found;
}

std::vector<cv::Rect> GetEyes(std::string HaarCascade, cv::Mat img, int minSize, int maxSize)
{
	cv::CascadeClassifier detector;
	std::string cascadeName = HaarCascade;;
	bool loaded = detector.load(cascadeName);
	// Parameters of detectMultiscale Cascade Classifier
	int groundThreshold = 1;
	double scaleStep = 1.1;
	cv::Size minimalObjectSize(minSize, minSize);
	cv::Size maximalObjectSize(maxSize, maxSize);

	// Vector of returned faces
	std::vector<cv::Rect> found;

	cv::Mat image_grey;
	found.clear();
	cvtColor(img, image_grey, CV_BGR2GRAY);
	// Detect faces
	detector.detectMultiScale(image_grey, found, scaleStep, groundThreshold, 0 | 2, minimalObjectSize, maximalObjectSize);
	// Draw the results into mat retrieved from webcam
	return found;
}
