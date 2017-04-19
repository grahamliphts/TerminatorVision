#include "classifiersClass.h"

Classifiers::Classifiers(std::string eye, std::string eyeGlass, std::string noze, std::string mouth, std::string face, std::string faceSide, std::string smile)
{
	eyeClassifierGlasses.load(eyeGlass);
	eyeClassifierDefault.load(eye);
	nozeClassifier.load(noze);
	mouthClassifier.load(mouth);
	faceClassifier.load(face);
	faceSideClassifier.load(faceSide);
	smileClassifer.load(smile);
}
