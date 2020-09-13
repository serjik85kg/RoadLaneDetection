#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

///////////////////////////////////////
// Debug draw functions step by step //
///////////////////////////////////////
namespace debugdraw
{
	void showImageReformerBinary(const cv::Mat& image);

	void showWarpedFitPolynomial(const cv::Mat& image);

	void showPerspectiveLinearLines(const cv::Mat& image);

	void showPerspectiveRectangle(const cv::Mat& image, int rectTopWidth = 250);

	void showWarpedAroundPolyArea(const cv::Mat& image);

	void showFinalSingleResult(const cv::Mat& image);
}