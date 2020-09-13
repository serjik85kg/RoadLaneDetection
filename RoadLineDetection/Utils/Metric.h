#pragma once
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

namespace metric
{
	// Calculater road radius and cars position
	// TO DO: delete radius (deprecated)
	std::tuple<double,double,double> measureCurvatureReal(cv::Size origImgSize, const std::vector<int>& leftFitX, const std::vector<int>& rightFitX, const cv::Mat& mInv);

	// Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
	cv::Mat drawInfoTxt(cv::Mat& img, const std::tuple<double,double,double>& metric,
		bool isFound);
}