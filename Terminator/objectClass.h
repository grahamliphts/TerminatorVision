#pragma once
#include "pointClass.h"
#include <vector>
#include <opencv2\core\core.hpp>


class Object {
	public:
		std::vector<point> contour;
		point barycentre;
		cv::Rect outterRect;


};