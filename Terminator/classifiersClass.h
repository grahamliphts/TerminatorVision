#pragma once

#include <opencv2/objdetect.hpp>
class Classifiers
{
public:
	cv::CascadeClassifier eyeClassifierGlasses;
	cv::CascadeClassifier eyeClassifierDefault;
	cv::CascadeClassifier nozeClassifier;
	cv::CascadeClassifier mouthClassifier;
	cv::CascadeClassifier faceClassifier;
	cv::CascadeClassifier faceSideClassifier;
	cv::CascadeClassifier smileClassifer;

	Classifiers(std::string eye, std::string eyeGlass, std::string noze, std::string mouth, std::string face, std::string faceSide, std::string smile);
};