// Terminator.cpp: definit le point d'entree pour l'application console.
//

//#define VIDEOMODE

#include "terminator.h"
#include <cstdlib>
#include <time.h>
#include <fstream>
std::vector<int> lastObjectCount;
int indexName = 0;

int main()
{
#ifdef VIDEOMODE
	cv::VideoCapture cameraStream("video.avi");
#else
	
	cv::VideoCapture cameraStream(0);
#endif

	if (!cameraStream.isOpened())
	{
		std::cout << "cannot open camera or video File" << std::endl;

		cvWaitKey(0);
		return 0;
	}
	else
		update(cameraStream);

	return 0;
}

void draw(std::vector<Object> objectList, std::vector<Face> faceList, cv::Mat Graph, Vignette &vignetteManager, cv::Mat img)
{
	//using namespace cv;
	cv::Mat imgresult;
	cv::Mat rgb[3];
	split(img, rgb);

	cv::Mat rgbaImg[4] = { cv::Mat(img.size(),CV_8UC1,cv::Scalar(0)),cv::Mat(img.size(),CV_8UC1,cv::Scalar(0)),rgb[2],cv::Mat(img.size(),CV_8UC1,cv::Scalar(1)) };
	merge(rgbaImg, 4, imgresult);


	for each (Face curFace in faceList)
	{
		//rectangle(img, curFace.face.outterRect.br(), curFace.face.outterRect.tl(), cv::Scalar(0, 0, 0), 1, 8, 0); // Draw Face outter rect

		rectangle(img, curFace.leftEye.outterRect.br(), curFace.leftEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Left eye
		rectangle(img, curFace.rightEye.outterRect.br(), curFace.rightEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Right eye
		/*if (curFace.isSmile)
			rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 255, 255), 1, 8, 0); // Draw mouse
		else
			rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 0, 255), 1, 8, 0); // Draw mouse
		rectangle(img, curFace.noze.outterRect.br(), curFace.noze.outterRect.tl(), cv::Scalar(255, 0, 255), 1, 8, 0); // Draw mouse
		*/

	}
	bool draw = true;
	for (int i = 0; i < objectList.size(); i++)
	{
		draw = true;
		Object blob = objectList[i];
		if (blob.outterRect.x > 0 && blob.outterRect.y > 0 && blob.outterRect.x + blob.outterRect.width < img.cols && blob.outterRect.y + blob.outterRect.height < img.rows)
		{
			for each (Face face in faceList)
			{
				draw = (((blob.outterRect & face.face.outterRect).area()) > 0);
				draw = !draw;
				;
			}
			if (draw)
			{
				cv::Mat subImage(img, cv::Rect(blob.outterRect.x, blob.outterRect.y, blob.outterRect.width, blob.outterRect.height));
				cv::Mat subImageGray(subImage);
				cv::GaussianBlur(subImageGray, subImageGray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

				/// Convert it to gray
				cvtColor(subImageGray, subImageGray, cv::COLOR_RGB2GRAY);

				/*
				/// Generate grad_x and grad_y
				cv::Mat grad;
				cv::Mat grad_x, grad_y;
				cv::Mat abs_grad_x, abs_grad_y;

				int scale = 2;
				int delta = 0;
				int ddepth = CV_16S;

				Sobel(subImageGray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
				convertScaleAbs(grad_x, abs_grad_x);

				Sobel(subImageGray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
				convertScaleAbs(grad_y, abs_grad_y);

				addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);*/

				cv::Canny(subImageGray, subImageGray, 20, 120, 3);

				cv::Mat dst;
				cv::Mat rgba[4] = { subImageGray,subImageGray,subImageGray,subImageGray };
				merge(rgba, 4, dst);
				//cv::GaussianBlur(dst, dst, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

				imgresult = OverlayImage(imgresult, dst, cv::Point(blob.outterRect.x, blob.outterRect.y));
				rectangle(imgresult, blob.outterRect, cv::Scalar(255, 255, 255));
			}
		}
	}

	for (int i = 0; i < faceList.size(); i++)
	{
		Object blob = faceList[i].face;

		cv::Mat subImage(img, cv::Rect(blob.outterRect.x, blob.outterRect.y, blob.outterRect.width, blob.outterRect.height));
		cv::Mat subImageGray(subImage);
		cv::GaussianBlur(subImageGray, subImageGray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

		/// Convert it to gray
		cvtColor(subImageGray, subImageGray, cv::COLOR_RGB2GRAY);

		/// Generate grad_x and grad_y
		cv::Mat grad;
		cv::Mat grad_x, grad_y;
		cv::Mat abs_grad_x, abs_grad_y;

		int scale = 2;
		int delta = 0;
		int ddepth = CV_16S;

		Sobel(subImageGray, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_x, abs_grad_x);

		Sobel(subImageGray, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
		convertScaleAbs(grad_y, abs_grad_y);

		addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

		cv::Mat dst;
		cv::Mat rgba[4] = { grad,grad,grad,grad };
		merge(rgba, 4, dst);

		imgresult = OverlayImage(imgresult, dst, cv::Point(blob.outterRect.x, blob.outterRect.y));
		rectangle(imgresult, blob.outterRect, cv::Scalar(255, 255, 255));
	}


	//vignette.copyTo(imgresult(cv::Rect(10, 10, vignette.cols, vignette.rows)));
	std::string randomName[] = { "Robert", "Luna MoonSilver", "John Cena" };

	int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
	int fontScale = 1;

	if (vignetteManager.changedFace)
	{
		int randomNumber = rand() % 3;
		while (randomNumber == indexName)
			randomNumber = rand() % 3;

		indexName = randomNumber;
		vignetteManager.changedFace = false;
	}
	cv::String name = "Name : " + randomName[indexName];
	cv::putText(imgresult, name, cv::Point(20 + vignetteManager.size, 40), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);

	if( faceList.size() >0 &&  faceList[0].isSmile)
		cv::putText(imgresult, "Target Happy", cv::Point(20 + vignetteManager.size, 60), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);
	else
		cv::putText(imgresult, "** Analyzing feels **", cv::Point(20 + vignetteManager.size, 60), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);

	for (int line = 0; line < 5; line++)
	{
		cv::String useless;
		for (int it = 0; it < 10; it++) {
			int randomval = rand() % 2;
			if (randomval == 0)
				useless += "1";
			else
				useless += "0";
		}
		cv::putText(imgresult, useless, cv::Point(0, (img.rows / 2) + (20*line)), fontFace, fontScale, cv::Scalar::all(255), 1, CV_AA);
	}

	//Graph
	auto pipVect = GetGraphSplitChannels(img);
	for (cv::Mat mat : pipVect)
	{
		cv::Mat tmp(mat.size() / 3, CV_8UC4);
		resize(mat, tmp, cv::Size(tmp.size()));
		tmp.copyTo(imgresult(cv::Rect(imgresult.cols - tmp.cols, imgresult.rows - tmp.rows, tmp.cols, tmp.rows)), tmp);
	}

	//Graph obj
	int interval;
	auto pip = GetGraphObjects(objectList.size(), interval);
	cv::Mat tmp(pip.size() / 3, CV_8UC4);
	resize(pip, tmp, cv::Size(tmp.size()));
	tmp.copyTo(imgresult(cv::Rect(0, imgresult.rows - tmp.rows, tmp.cols, tmp.rows)), tmp);
	cv::Point objCounter((interval/3) * 10, imgresult.size().height - objectList.size() * 5 );
	cv::putText(imgresult, std::to_string(objectList.size()), objCounter, fontFace, fontScale, cv::Scalar::all(255));

	//cv::resize(imgresult, imgresult, cv::Size(imgresult.cols  * 2 , imgresult.rows * 2));
	imshow("Terminator Vision", imgresult);
	//cv::resizeWindow("Terminator Vision", 800, 800);
	//int key2 = cv::waitKey(20);
}

void update(cv::VideoCapture cameraStream)
{
	cv::Mat currentImg;
	std::vector<Object> objects;
	std::vector<Face> faces;
	cv::Mat graph;

	cv::Mat cameraFrame;
	Classifiers HaarManager("HaarCascade/haarcascade_eye.xml",
		"HaarCascade/haarcascade_eye_tree_eyeglasses.xml",
		"HaarCascade/haarcascade_mcs_nose.xml",
		"HaarCascade/haarcascade_mcs_mouth.xml",
		"HaarCascade/haarcascade_frontalface_default.xml",
		"HaarCascade/haarcascade_profileface.xml",
		"HaarCascade/haarcascade_smile.xml");

	//extract img to create first vignette
	cameraStream.read(currentImg);
	Vignette vignetteManager(currentImg);

	while (true) 
	{
		cameraStream.read(currentImg);
		//currentImg = cv::imread("john-cena.jpg", CV_LOAD_IMAGE_COLOR);
		//currentImg = cv::imread("Terminator_metal.jpg", CV_LOAD_IMAGE_COLOR);
		//imshow("cam", currentImg);

		objects = ObjectDetection(currentImg);
		faces = FaceDetection(currentImg, &HaarManager);
		//graph = GetGraph(currentImg);
		vignetteManager.Process(currentImg, faces);
		draw(objects, faces, graph, vignetteManager, currentImg);
		//	cv::waitKey(0);
		if (cv::waitKey(30) >= 0)
			break;

	}
}







