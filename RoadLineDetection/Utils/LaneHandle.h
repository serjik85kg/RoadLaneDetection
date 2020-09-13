#pragma once
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>

namespace laneHandle
{
	namespace transforms
	{
		// Get default src-dst warp transformation points //
		std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getDefaultSrcDstCorners(const int imageW, const int imageH);

		// Calculation two transformation Matrixs //
		std::pair<cv::Mat, cv::Mat> calcPerspectiveMat(const std::vector<cv::Point2f>& srcCorners, const std::vector<cv::Point2f>& dstCorners); 

		// Get warped image //
		cv::Mat perspectiveTransform(const cv::Mat& image, const cv::InputArray& transformMat); 

		// Get one of 2 warped images //
		cv::Mat getWarped(const cv::Mat& img, int flag); // TO DO: remove and replace with perspectiveTransform

		// Get default matrix transformations //
		std::pair<cv::Mat, cv::Mat> getDefaultGM(const cv::Mat& image);
	}

	//////////////////////
	//Detect Lane Pixels//
	//////////////////////
	std::tuple<std::vector<int>, std::vector<int>, 
		std::vector<int>, std::vector<int>, cv::Mat> findLanePixels(const cv::Mat& binaryWarped, bool isShow);

	// Use polyfit function on our points //
	std::tuple<int, std::vector<float>, std::vector<float>, std::vector<cv::Point>, std::vector<cv::Point>, cv::Mat>
		fitPolynomial(const cv::Mat& binaryWarped, bool isShow);

	///////////////////////////////////
	// Find linear perspective lines //
	///////////////////////////////////
	// Additional help function //
	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getDotsPair(
		const std::vector<cv::Point>& leftX, const std::vector<cv::Point>& rightX, const cv::Mat& debugImg);

	// Input image must be not warped //
	std::pair<std::pair<float, float>, std::pair<float, float>> findLinearLines(
		const std::vector<cv::Point2f>& leftDots, const std::vector<cv::Point2f>& rightDots,
		cv::Mat& dbgImg, bool isShow);

	// Find perspective rect //
	std::vector<cv::Point> findPerspectiveRect(const std::pair<float, float> leftLineCoeffs,
		std::pair<float, float> rightLineCoeffs, int rectTopWidth, cv::Mat& dbgImg, bool isShow);

	// Correction of polynomial function (check around pixels) //
	std::pair<std::vector<float>, std::vector<float>> fitPoly(const std::vector<int>& leftX,
		const std::vector<int>& leftY, const std::vector<int>& rightX, const std::vector<int>& rightY);
	std::tuple<std::vector<int>, std::vector<int>, float> findPolyValues(const std::vector<float>& leftFit,
		const std::vector<float>& rightFit, cv::Size imageSize);
	std::tuple<int, std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>, cv::Mat>
		findAroundPoly(const cv::Mat& binaryWarped, const std::vector<float>& leftFit, const std::vector<float>& rightFit);

	// Unwrap and draw the lane on the original image //
	cv::Mat drawLane(const cv::Mat& image, const cv::Mat& warped, const std::vector<int>& lfx, const std::vector<int>& rfx, const cv::Mat& gmInv);

}