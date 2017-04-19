// Terminator.cpp : définit le point d'entrée pour l'application console.
//

#include "terminator.h"

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
		"haarcascade_mcs_nose.xml",
		"HaarCascade/haarcascade_mcs_mouth.xml",
		"HaarCascade/haarcascade_frontalface_default.xml",
		"HaarCascade/haarcascade_profileface.xml",
		"HaarCascade/haarcascade_smile.xml");

	while(true){
		cameraStream.read(currentImg);
		//currentImg = cv::imread("john-cena.jpg", CV_LOAD_IMAGE_COLOR);
		imshow("cam", currentImg);

		objects = ObjectDetection(currentImg);
		faces = FaceDetection(currentImg,&HaarManager);
		graph = GetGraph(currentImg);
		vignette = GetVignette(currentImg, faces);
	//	cv::waitKey(0);
		if (cv::waitKey(30) >= 0)
			break;
	
	}
}

void draw(std::vector<Object> objectList, std::vector<Face> faceList, cv::Mat Graph,cv::Mat* img)
{

	

}





