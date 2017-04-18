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

	cv::Mat cameraFrame;

	while(true){
		cameraStream.read(currentImg);
		imshow("cam", currentImg);

		objects = ObjectDetection(currentImg);
		faces = FaceDetection(currentImg);
		graph = GetGraph(currentImg);

		if (cv::waitKey(30) >= 0)
			break;
	}


}

void draw(std::vector<Object> objectList, std::vector<Face> faceList, cv::Mat Graph,cv::Mat* img)
{

	

}





