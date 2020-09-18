#include "Debugdraw.h"

#include "../Utils/ImageReformer.h"
#include "../Utils/LaneHandle.h"
#include "../Utils/Metric.h"

namespace debugdraw
{
	void showImageReformerBinary(const cv::Mat& image)
	{
		auto binaryOuts = imageReformer::makeBinary(image);
		auto combinedBinary = binaryOuts[0];
		auto coloredBinary = binaryOuts[1];
		cv::imshow("coloredBinaryImage", combinedBinary);
		cv::waitKey();
	}

	void showWarpedFitPolynomial(const cv::Mat& image)
	{
		auto binaryOuts = imageReformer::makeBinary(image);
		auto& combinedBinary = binaryOuts[0];
		auto warped = laneHandle::transforms::getWarped(combinedBinary, 0);
		auto[height, lfit, rfit, lfx, rfx, showImg] = laneHandle::fitPolynomial(warped, true);
		cv::imshow("WarpedFitPolynomial", showImg);
		cv::waitKey();
	}

	void showPerspectiveLinearLines(const cv::Mat& image)
	{
		auto binaryOuts = imageReformer::makeBinary(image);
		auto& combinedBinary = binaryOuts[0];
		auto warped = laneHandle::transforms::getWarped(combinedBinary, 0);
		auto[height, lfit, rfit, lfx, rfx, _none] = laneHandle::fitPolynomial(warped, true);
		std::vector<cv::Mat> dbgVec{ combinedBinary,combinedBinary,combinedBinary };
		cv::Mat showImg;
		cv::merge(dbgVec, showImg);
		auto[leftDots, rightDots] = laneHandle::getDotsPair(lfx, rfx, warped);
		auto[leftL, rightL] = laneHandle::findLinearLines(leftDots, rightDots, showImg, true);
		cv::imshow("PerspectiveLines", showImg);
		cv::waitKey();
	}

	void showPerspectiveRectangle(const cv::Mat& image, int rectTopWidth)
	{
		auto binaryOuts = imageReformer::makeBinary(image);
		auto& combinedBinary = binaryOuts[0];
		auto warped = laneHandle::transforms::getWarped(combinedBinary, 0);
		auto[height, lfit, rfit, lfx, rfx, _none] = laneHandle::fitPolynomial(warped, true);
		std::vector<cv::Mat> dbgVec{ combinedBinary,combinedBinary,combinedBinary };
		cv::Mat showImg;
		cv::merge(dbgVec, showImg);
		auto[leftDots, rightDots] = laneHandle::getDotsPair(lfx, rfx, warped);
		auto[leftL, rightL] = laneHandle::findLinearLines(leftDots, rightDots, showImg, false);
		auto rect = laneHandle::findPerspectiveRect(leftL, rightL, rectTopWidth, showImg, true);
		cv::imshow("PerspectiveRect", showImg);
		cv::waitKey();
	}

	void showWarpedAroundPolyArea(const cv::Mat& image)
	{
		auto binaryOuts = imageReformer::makeBinary(image);
		auto& combinedBinary = binaryOuts[0];
		auto warped = laneHandle::transforms::getWarped(combinedBinary, 0);
		auto[height, lfit, rfit, lfx, rfx, _none] = laneHandle::fitPolynomial(warped, true);
		auto[plt, lf, rf, newlfix, newrfix, warpedPoly] = laneHandle::findAroundPoly(warped, lfit, rfit);
		cv::imshow("Warped Around Poly", warpedPoly);
		cv::waitKey();
	}

	void showFinalSingleResult(const cv::Mat& image)
	{
		auto binaryOuts = imageReformer::makeBinary(image);
		auto& combinedBinary = binaryOuts[0];
		auto warped = laneHandle::transforms::getWarped(combinedBinary, 0);
		auto[height, lfit, rfit, lfx, rfx, _none] = laneHandle::fitPolynomial(warped, true);
		auto[plt, lf, rf, newlfix, newrfix, warpedPoly] = laneHandle::findAroundPoly(warped, lfit, rfit);
		const auto mInv = laneHandle::transforms::getDefaultGM(image).second;
		cv::Mat output = laneHandle::drawLane(image, warpedPoly, newlfix, newrfix, mInv);
		auto metricTuple = metric::measureCurvatureReal(image.size(), newlfix, newrfix, mInv);
		output = metric::drawInfoTxt(output, metricTuple, true);
		cv::imshow("Result", output);
		cv::waitKey();
	}
}