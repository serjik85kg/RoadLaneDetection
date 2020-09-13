#include "Metric.h"
#include "EigenOperations.h"

namespace metric
{
	std::tuple<double, double, double> measureCurvatureReal(cv::Size origImageSize, const std::vector<int>& leftFitX, const std::vector<int>& rightFitX, const cv::Mat& mInv)
	{
		int height = origImageSize.height;
		int width = origImageSize.width;
		int yEval = origImageSize.height;

		auto ymPerPix = 30. / 720;
		auto xmPerPix = 3.7 / 700;

		std::vector<int> yVec(height);
		for (auto& p : yVec)
		{
			p = ymPerPix * (&p - &(*std::begin(yVec)));
		}
		std::vector<int> leftXVec(height);
		std::vector<int> rightXVec(height);
		// TO DO: check speed here
		for (auto& p : leftXVec)
		{
			int index = &p - &(*std::begin(leftXVec));
			p = leftFitX[index] * xmPerPix;
			// not nice but effective
			rightXVec[index] = rightFitX[index] * xmPerPix;
		}

		auto leftFitCr = eigenOperations::polyfit(yVec, leftXVec, 2);
		auto rightFitCr = eigenOperations::polyfit(yVec, rightXVec, 2);


		// y: Ax^2 + Bx + C
		// R = (1 + y'^2)^(1.5)/y''
		auto ex1 = 2 * leftFitCr[2] * yEval * ymPerPix + leftFitCr[1];
		auto leftCurverad = (std::pow(1 + std::pow(ex1, 2), 1.5)) / abs(2 * leftFitCr[2]);
		auto ex2 = 2 * rightFitCr[2] * yEval * ymPerPix + rightFitCr[1];
		auto rightCurverad = (std::pow(1 + std::pow(ex2, 2), 1.5)) / abs(2 * rightFitCr[2]);

		std::vector<cv::Point2f> invDots{ cv::Point2f(leftFitX[height - 1], height), cv::Point2f(rightFitX[height - 1], height) };
		std::vector<cv::Point2f> origDots;
		cv::perspectiveTransform(invDots, origDots, mInv);

		xmPerPix = 3.7 / abs(origDots[0].x - origDots[1].x);
		auto pos = (width / 2 - (origDots[0].x + origDots[1].x) / 2) * xmPerPix;
		
		return std::make_tuple(leftCurverad, rightCurverad, pos);
	}

	// Add std optional
	cv::Mat drawInfoTxt(cv::Mat& img, const std::tuple<double, double, double>& metric,
		bool isFound)
	{
		auto font = cv::FONT_HERSHEY_DUPLEX;
		auto[leftCurveRad, rightCurveRad, relativePos] = metric;
		auto curve = (leftCurveRad + rightCurveRad) / 2;
		//cv::putText(img, "Radius: " + std::to_string(curve) + " m", cv::Point(50, 50), font, 0.8, { 0,255,0 }, 2, cv::LINE_AA);
		cv::putText(img, "Position: " + std::to_string(relativePos) + " m", cv::Point(50, 90), font, 0.8, { 0,255,0 }, 2, cv::LINE_AA);
		return img;
	}
}