// Terminator.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"
#include <ostream>
#include <iostream>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"

int main()
{
	cv::Mat img;
	img = cv::imread("Terminator.jpg");
	cv::imshow("Terminator", img);
	cv::waitKey();
    return 0;
}

