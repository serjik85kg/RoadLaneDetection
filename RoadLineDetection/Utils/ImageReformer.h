#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

namespace imageReformer {

	std::vector<cv::Mat> makeBinary(const cv::Mat& img);
	// TO DO: add simplier makeBinary
	// TO DO: add floodfill road detector
};