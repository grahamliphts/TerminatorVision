#include "graphCreation.h"
#include <opencv2/highgui/highgui.hpp>

cv::Mat GetGraph(cv::Mat img, int step, int height, int width)
{
	/// Separate the image in 3 places ( B, G and R )
	std::vector<cv::Mat> bgr_planes;
	cv::split(img, bgr_planes);

	/// Establish the number of bins
	auto histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	auto uniform = true;
	auto accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	/// Compute the histograms
	calcHist(&bgr_planes[0], 1, nullptr, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, nullptr, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, nullptr, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Draw the histograms for B, G and R
	int hist_w = width;
	int hist_h = height;
	int bin_w = cvRound(static_cast<double>(hist_w) / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

	/// Draw for each channel
	for (int i = step; i < histSize; i += step)
	{
		//create rectangles
		auto pt1 = cv::Point(bin_w * (i - step), hist_h);
		auto pt2 = cv::Point(bin_w * i, hist_h);

		auto pt3 = cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i)));
		auto pt4 = cv::Point(bin_w * (i - step), hist_h - cvRound(b_hist.at<float>(i - step)));
		cv::Point ptsB[] = { pt1, pt2, pt3, pt4, pt1 };
		fillConvexPoly(histImage, ptsB, 5, cv::Scalar(255, 0, 0, 1));

		pt3 = cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i)));
		pt4 = cv::Point(bin_w * (i - step), hist_h - cvRound(g_hist.at<float>(i - step)));
		cv::Point ptsG[] = { pt1, pt2, pt3, pt4, pt1 };
		fillConvexPoly(histImage, ptsG, 5, cv::Scalar(0, 255, 0, 1));

		pt3 = cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i)));
		pt4 = cv::Point(bin_w * (i - step), hist_h - cvRound(r_hist.at<float>(i - step)));
		cv::Point ptsR[] = { pt1, pt2, pt3, pt4, pt1 };
		fillConvexPoly(histImage, ptsR, 5, cv::Scalar(0, 0, 255, 1));

		//create lines
		line(histImage, cv::Point(bin_w*(i - step), hist_h - cvRound(b_hist.at<float>(i - step))),
			cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 125, 0, 1), 2, 8, 0);
		line(histImage, cv::Point(bin_w*(i - step), hist_h - cvRound(g_hist.at<float>(i - step))),
			cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(125, 255, 0, 1), 2, 8, 0);
		line(histImage, cv::Point(bin_w*(i - step), hist_h - cvRound(r_hist.at<float>(i - step))),
			cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 125, 255, 1), 2, 8, 0);
	}

	/// Display
	//cv::namedWindow("Graph", CV_WINDOW_NORMAL);
	//imshow("Graph", histImage);

	return histImage;
}

std::vector<cv::Mat> GetGraphSplitChannels(cv::Mat img, int step, int height, int width)
{
	/// Separate the image in 3 places ( B, G and R )
	std::vector<cv::Mat> bgr_planes;
	cv::split(img, bgr_planes);

	/// Establish the number of bins
	auto histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	auto uniform = true;
	auto accumulate = false;

	cv::Mat b_hist, g_hist, r_hist;

	/// Compute the histograms
	calcHist(&bgr_planes[0], 1, nullptr, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, nullptr, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, nullptr, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Draw the histograms for B, G and R
	int hist_w = width;
	int hist_h = height;
	int bin_w = cvRound(static_cast<double>(hist_w) / histSize);

	cv::Mat histImageB(hist_h, hist_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
	cv::Mat histImageG(hist_h, hist_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));
	cv::Mat histImageR(hist_h, hist_w, CV_8UC4, cv::Scalar(0, 0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImageB.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(g_hist, g_hist, 0, histImageG.rows, cv::NORM_MINMAX, -1, cv::Mat());
	normalize(r_hist, r_hist, 0, histImageR.rows, cv::NORM_MINMAX, -1, cv::Mat());

	/// Draw for each channel
	for (int i = step; i < histSize; i += step)
	{
		////create rectangles
		//auto pt1 = cv::Point(bin_w * (i - step), hist_h);
		//auto pt2 = cv::Point(bin_w * i, hist_h);

		//auto pt3 = cv::Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i)));
		//auto pt4 = cv::Point(bin_w * (i - step), hist_h - cvRound(b_hist.at<float>(i - step)));
		//cv::Point ptsB[] = { pt1, pt2, pt3, pt4, pt1 };
		//fillConvexPoly(histImageB, ptsB, 5, cv::Scalar(255, 255, 255));

		//pt3 = cv::Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i)));
		//pt4 = cv::Point(bin_w * (i - step), hist_h - cvRound(g_hist.at<float>(i - step)));
		//cv::Point ptsG[] = { pt1, pt2, pt3, pt4, pt1 };
		//fillConvexPoly(histImageG, ptsG, 5, cv::Scalar(255, 255, 255));

		//pt3 = cv::Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i)));
		//pt4 = cv::Point(bin_w * (i - step), hist_h - cvRound(r_hist.at<float>(i - step)));
		//cv::Point ptsR[] = { pt1, pt2, pt3, pt4, pt1 };
		//fillConvexPoly(histImageR, ptsR, 5, cv::Scalar(255, 255, 255));

		//create lines
		line(histImageB, cv::Point(bin_w*(i - step), hist_h - cvRound(b_hist.at<float>(i - step))),
			cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 255, 255), 8, 8, 0);
		line(histImageG, cv::Point(bin_w*(i - step), hist_h - cvRound(g_hist.at<float>(i - step))),
			cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(255, 255, 255), 8, 8, 0);
		line(histImageR, cv::Point(bin_w*(i - step), hist_h - cvRound(r_hist.at<float>(i - step))),
			cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(255, 255, 255), 8, 8, 0);
	}

	const int dist = 50;
	cv::Mat grid(hist_h, hist_w, CV_8UC4, cv::Scalar(0, 0, 0));

	for (int i = 0; i < height; i += dist)
		cv::line(grid, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 255, 255), 4);
	for (int i = 0; i < width; i += dist)
		cv::line(grid, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 255, 255), 4);

	//std::vector<cv::Mat> data = { histImageB, histImageG, histImageR };

	return { grid, histImageB, histImageG, histImageR };
}

cv::Mat RedPicture(cv::Mat img)
{
	cv::Mat black(img.size(), CV_8UC1);
	black.setTo(0);

	//split
	std::vector<cv::Mat> channels;
	split(img, channels);

	std::vector<cv::Mat> out_chan;
	out_chan.push_back(black);
	out_chan.push_back(black);
	out_chan.push_back(channels[2]);
	cv::Mat outR(img.size(), CV_8UC3);
	merge(out_chan, outR);

	imshow("R", outR);

	return outR;
}

cv::Mat GetGraphObjects(int objCount, int height, int width)
{
	cv::Mat out(height, width, CV_8UC4);
	out.setTo(0);

	if (lastObjectCount.size() == 10)
		lastObjectCount.erase(lastObjectCount.begin());

	lastObjectCount.push_back(objCount);
	int intervalY = width / 10;

	for (int i = 0; i < height; i += intervalY)
		cv::line(out, cv::Point(0, i), cv::Point(width, i), cv::Scalar(255, 255, 255), 4);
	for (int i = 0; i < width; i += intervalY)
		cv::line(out, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 255, 255), 4);

	for (int i = 1; i< lastObjectCount.size(); ++i)
	{
		/*auto pt1 = cv::Point(intervalY*(i - 1), 0);
		auto pt2 = cv::Point(intervalY * i, 0);

		auto pt3 = cv::Point(intervalY * i, lastObjectCount[i] * 10);
		auto pt4 = cv::Point(intervalY*(i - 1), lastObjectCount[i - 1] * 10);
		cv::Point pts[] = { pt1, pt2, pt3, pt4, pt1 };
		fillConvexPoly(out, pts, 5, cv::Scalar(255, 255, 255));*/

		cv::line(out, cv::Point(intervalY*(i - 1), lastObjectCount[i - 1] * 10), cv::Point(intervalY*i, lastObjectCount[i] * 10), cv::Scalar(255, 255, 255), 8, 8, 0);

		
	}

	//cv::imshow("ObjCount", out);

	return out;
}

// deprecated
void connectAll(cv::Mat img, int objCount)
{
	cv::Mat out(img);
	auto pipVect = GetGraphSplitChannels(img);



	for (cv::Mat mat : pipVect)
	{
		cv::Mat tmp(mat.size()/2, CV_8UC3);
		resize(mat, tmp, cv::Size(tmp.size()));
		tmp.copyTo(out(cv::Rect(out.cols - tmp.cols, out.rows - tmp.rows, tmp.cols, tmp.rows)), tmp);
	}

	auto pip = GetGraphObjects(objCount);
	cv::Mat tmp(pip.size() / 2, CV_8UC3);
	resize(pip, tmp, cv::Size(tmp.size()));
	tmp.copyTo(out(cv::Rect(0, out.rows - tmp.rows, tmp.cols, tmp.rows)), tmp);

	imshow("effect", out);
}
