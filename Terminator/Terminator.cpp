// Terminator.cpp: definit le point d'entree pour l'application console.
//

#include "terminator.h"

std::vector<int> lastObjectCount;

int main()
{

	cv::VideoCapture cameraStream(0);


	if (!cameraStream.isOpened())
	{
		std::cout << "cannot open camera";
		
		cvWaitKey(0);
		return 0;
	}
	else
		update(cameraStream);

	return 0;
}

void draw(std::vector<Object> objectList, std::vector<Face> faceList, cv::Mat Graph,cv::Mat vignette, cv::Mat img)
{
	//using namespace cv;
	cv::Mat rgb[3];
	split(img, rgb);

	cv::Mat rgbaImg[4] = { cv::Mat(img.size(),CV_8UC1,cv::Scalar(0)),cv::Mat(img.size(),CV_8UC1,cv::Scalar(0)),rgb[2],cv::Mat(img.size(),CV_8UC1,cv::Scalar(1)) };
	merge(rgbaImg, 4, img);

	
	for each (Face curFace in faceList)
	{
		//rectangle(img, curFace.face.outterRect.br(), curFace.face.outterRect.tl(), cv::Scalar(0, 0, 0), 1, 8, 0); // Draw Face outter rect
		rectangle(img, curFace.leftEye.outterRect.br(), curFace.leftEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Left eye
		rectangle(img, curFace.rightEye.outterRect.br(), curFace.rightEye.outterRect.tl(), cv::Scalar(0, 255, 0), 1, 8, 0); // Draw Right eye
		if (curFace.isSmile)
			rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 255, 255), 1, 8, 0); // Draw mouse
		else
			rectangle(img, curFace.mouse.outterRect.br(), curFace.mouse.outterRect.tl(), cv::Scalar(0, 0, 255), 1, 8, 0); // Draw mouse
		rectangle(img, curFace.noze.outterRect.br(), curFace.noze.outterRect.tl(), cv::Scalar(255, 0, 255), 1, 8, 0); // Draw mouse

	}

	for (int i = 0; i < objectList.size(); i++)
	{
		Object blob = objectList[i];
		if (blob.outterRect.x > 0 && blob.outterRect.y > 0 && blob.outterRect.x + blob.outterRect.width < img.cols && blob.outterRect.y + blob.outterRect.height < img.rows)
		{
			for each (Face face in faceList)
			{

			}
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

			img = OverlayImage(img, dst, cv::Point(blob.outterRect.x, blob.outterRect.y));
			rectangle(img, blob.outterRect, cv::Scalar(255, 255, 255));
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

		img = OverlayImage(img, dst, cv::Point(blob.outterRect.x, blob.outterRect.y));
		rectangle(img, blob.outterRect, cv::Scalar(255, 255, 255));
	}

	
	imshow("FaceDetected", img);
	//int key2 = cv::waitKey(20);
}

void update(cv::VideoCapture cameraStream)
{
	cv::Mat currentImg;
	std::vector<Object> objects;
	std::vector<Face> faces;
	cv::Mat graph;
	cv::Mat vignette;

	cv::Mat cameraFrame;
	Classifiers HaarManager("HaarCascade/haarcascade_eye.xml",
		"HaarCascade/haarcascade_eye_tree_eyeglasses.xml",
		"HaarCascade/haarcascade_mcs_nose.xml",
		"HaarCascade/haarcascade_mcs_mouth.xml",
		"HaarCascade/haarcascade_frontalface_default.xml",
		"HaarCascade/haarcascade_profileface.xml",
		"HaarCascade/haarcascade_smile.xml");


	while(true){
		cameraStream.read(currentImg);
		currentImg = cv::imread("john-cena.jpg", CV_LOAD_IMAGE_COLOR);
		//currentImg = cv::imread("Terminator_metal.jpg", CV_LOAD_IMAGE_COLOR);
		//imshow("cam", currentImg);

		objects = ObjectDetection(currentImg);
		faces = FaceDetection(currentImg,&HaarManager);
		//graph = GetGraph(currentImg);
		//vignette = GetVignette(currentImg, faces);
		draw(objects, faces, graph, vignette, currentImg);
	//	cv::waitKey(0);
		if (cv::waitKey(30) >= 0)
			break;
	
	}
}







