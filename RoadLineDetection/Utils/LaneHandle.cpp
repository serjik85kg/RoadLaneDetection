#include "LaneHandle.h"
#include "EigenOperations.h"

namespace laneHandle
{
	namespace transforms
	{

		std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getDefaultSrcDstCorners(const int imageW, const int imageH) //1280, 720
		{
			// Constants for any image scale
			const float leftDownCoefX = 0.171875;
			const float leftUpCoefX = 0.453125;
			const float rightUpCoefX = 0.546875;
			const float rightDownCoefX = 0.875;
			const float upCoefY = 0.6388888;

			const float leftDstCoefX = 0.21875;
			const float rightDstCoefX = 0.78125;

			std::vector<cv::Point2f> srcCorners =
			{ cv::Point2f(leftDownCoefX * imageW, imageH), cv::Point2f(leftUpCoefX * imageW, upCoefY * imageH), 
				cv::Point2f(rightUpCoefX * imageW, upCoefY * imageH), cv::Point2f(rightDownCoefX * imageW, imageH) };
			std::vector<cv::Point2f> dstCorners =
			{ cv::Point2f(leftDstCoefX * imageW, imageH), cv::Point2f(leftDstCoefX * imageW, 0), 
				cv::Point2f(rightDstCoefX * imageW, 0), cv::Point2f(rightDstCoefX * imageW, imageH) };

			return std::make_pair(srcCorners, dstCorners);
		}

		std::pair<cv::Mat, cv::Mat> calcPerspectiveMat(const std::vector<cv::Point2f>& srcCorners, const std::vector<cv::Point2f>& dstCorners)
		{
			auto m = cv::getPerspectiveTransform(srcCorners, dstCorners);
			auto mInv = cv::getPerspectiveTransform(dstCorners, srcCorners);
			return std::make_pair(m, mInv);
		}

		cv::Mat perspectiveTransform(const cv::Mat& image, const cv::InputArray& transformMat)
		{
			cv::Mat warped;
			cv::warpPerspective(image, warped, transformMat, cv::Size(image.cols, image.rows), cv::INTER_LINEAR);
			return warped;
		}

		cv::Mat getWarped(const cv::Mat& img, int flag)
		{
			int imageW = img.cols;
			int imageH = img.rows;
			auto cornersSrcDst = getDefaultSrcDstCorners(imageW, imageH);
			auto transMatPair = calcPerspectiveMat(cornersSrcDst.first, cornersSrcDst.second);
			cv::Mat warpedImg;
			switch (flag) {
			case 0:
				warpedImg = perspectiveTransform(img, transMatPair.first);
				break;
			case 1:
				warpedImg = perspectiveTransform(img, transMatPair.second);
				break;
			default:
				break;
			}
			return warpedImg;
		}

		std::pair<cv::Mat, cv::Mat> getDefaultGM(const cv::Mat& img)
		{
			int imageW = img.cols;
			int imageH = img.rows;
			auto [srcCorners, dstCorners] = getDefaultSrcDstCorners(imageW, imageH);
			auto transMatPair = calcPerspectiveMat(srcCorners, dstCorners);
			return transMatPair;
		}
	}


	// help functions for detect lane //
	namespace
	{
		// Get bottom half of image //
		cv::Mat getBottomHalf(const cv::Mat& image) {
			cv::Mat bottomHalf = image(cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));
			return bottomHalf;
		}
		// Get histogram of "white" pixels per cols //
		std::vector<int> getHistogram(const cv::Mat& binaryWarpedBottom)
		{
			assert(binaryWarpedBottom.channels() == 1);
			std::vector<int> histogram;
			histogram.reserve(binaryWarpedBottom.cols);
			for (size_t j = 0; j < binaryWarpedBottom.cols; ++j)
			{
				//const uchar * ptrj = binaryWarpedBottom.ptr<uchar>(j);
				int histElement = 0;
				for (size_t i = 0; i < binaryWarpedBottom.rows; ++i)
				{
					histElement += static_cast<int>(binaryWarpedBottom.at<uchar>(i, j));
				}
				histogram.emplace_back(histElement);
			}
			return histogram;
		}

		// Find medium x coordinate of the points array
		int findMediumCoordsX(const std::vector<cv::Point>& laneIdx)
		{
			double sum = 0;
			for (auto& p : laneIdx)
			{
				sum += p.x;
			}
			//for (size_t i = 0; i < laneIdx.size(); ++i)
			//	sum += laneIdx[i].x;
			int medium = sum / static_cast<float>(laneIdx.size());
			return medium;
		}
		// Deprecated
		int findMediumCoordsY(const std::vector<cv::Point>& laneIdx)
		{
			double sum = 0;
			for (auto& p : laneIdx)
			{
				sum += p.y;
			}
			//for (size_t i = 0; i < laneIdx.size(); ++i)
			//	sum += laneIdx[i].y;
			int medium = sum / static_cast<float>(laneIdx.size());
			return medium;
		}
	}

	// Detect left and right lane pixels //
	std::tuple<std::vector<int>, std::vector<int>, 
		std::vector<int>, std::vector<int>, cv::Mat> findLanePixels(const cv::Mat& binaryWarped, bool isShow)
	{
		// Visualisation
		cv::Mat showImg;
		if (isShow)
		{
			std::vector<cv::Mat> binaryWarpedVec{ binaryWarped,binaryWarped,binaryWarped };
			cv::merge(binaryWarpedVec, showImg);
		}
		/////////////////////////////////////////////////////////
		// Take a histogram of bottom half of the warped image //
		auto binaryWarpedBottom = getBottomHalf(binaryWarped);
		auto histogram = getHistogram(binaryWarpedBottom);
		// Find the peak of the left and right halves of histogram
		auto midpoint = histogram.size() / 2;
		auto leftXItMax = std::max_element(histogram.begin(), histogram.begin() + midpoint);
		auto rightXItMax = std::max_element(histogram.begin() + midpoint, histogram.end());
		int leftXBaseIdx = std::distance(histogram.begin(), leftXItMax);
		int rightXBaseIdx = std::distance(histogram.begin(), rightXItMax);
		/////////////////////////////////////////////////////////

		// Set the number of sliding windows (You can "play" with this parametr)
		int nwindows = 9; //
		// Set the width of the windows +/- margin //
		const float marginWidthCf = 0.078125;
		int margin = binaryWarped.cols * marginWidthCf; // 100 for 1280x720
		// Set minimum number of pixels found to recenter window
		const float minpixCoef = 0.009375;
		int windowHeight = binaryWarped.rows / nwindows;
		int minpix = minpixCoef * (windowHeight * 2 * margin); // 150 for 1280x720

		// Find nonzero pixels //
		cv::Mat nonzero;
		cv::findNonZero(binaryWarped, nonzero);
		std::vector<std::pair<int, int>> nonzeroXY(nonzero.total());
		for (size_t i = 0; i < nonzero.total(); ++i)
		{
			nonzeroXY[i] = std::make_pair(nonzero.at<cv::Point>(i).x, nonzero.at<cv::Point>(i).y);
		}

		// Current positions to be updated later for each window in nwindows //
		auto leftXCurrentIdx = leftXBaseIdx;
		auto rightXCurrentIdx = rightXBaseIdx;
		// Create empty lists to receive left and right lane pixel indices //
		std::vector<cv::Point> leftLaneIdx;
		std::vector<cv::Point> rightLaneIdx;

		// Step through the windows one by one //
		for (int iwindow = nwindows - 1; iwindow >= 0; --iwindow)
		{
			auto winYLow = iwindow * windowHeight;
			auto winYHigh = (iwindow + 1) * windowHeight;
			auto winXLeftLow = leftXCurrentIdx - margin;
			auto winXLeftHigh = leftXCurrentIdx + margin;
			auto winXRightLow = rightXCurrentIdx - margin;
			auto winXRightHigh = rightXCurrentIdx + margin;

			if (isShow)
			{
				cv::rectangle(showImg, cv::Point(winXLeftLow, winYLow),
					cv::Point(winXLeftHigh, winYHigh), { 0, 255, 0 }, 2);
				cv::rectangle(showImg, cv::Point(winXRightLow, winYLow),
					cv::Point(winXRightHigh, winYHigh), { 0, 255, 0 }, 2);
			}

			// Identify the nonzero pixels in x and y within the window //
			std::vector<cv::Point> goodLeftIdx;
			std::vector<cv::Point> goodRightIdx;
			goodLeftIdx.reserve(nonzeroXY.size());
			goodRightIdx.reserve(nonzeroXY.size());
			for (auto& p : nonzeroXY)
			{
				if ((p.second >= winYLow) && (p.second < winYHigh) &&
					(p.first >= winXLeftLow) && (p.first < winXLeftHigh))
				{
					goodLeftIdx.push_back(cv::Point(p.first, p.second));
				}
				if ((p.second >= winYLow) && (p.second < winYHigh) &&
					(p.first >= winXRightLow) && (p.first < winXRightHigh))
				{
					goodRightIdx.push_back(cv::Point(p.first, p.second));
				}
			}
			// Append these goodIdx to left and right idxs //
			leftLaneIdx.insert(leftLaneIdx.end(), goodLeftIdx.begin(), goodLeftIdx.end());
			rightLaneIdx.insert(rightLaneIdx.end(), goodRightIdx.begin(), goodRightIdx.end());

			// If found > minpix pixels, recenter next window //
			// (right or leftx current) on their mean position //
			if (goodLeftIdx.size() > minpix)
			{
				auto leftXMean = findMediumCoordsX(goodLeftIdx);
				leftXCurrentIdx = leftXMean;
			}
			if (goodRightIdx.size() > minpix)
			{
				auto rightXMean = findMediumCoordsX(goodRightIdx);
				rightXCurrentIdx = rightXMean;
			}
		}

		std::vector<int> leftX(leftLaneIdx.size());
		std::vector<int> leftY(leftLaneIdx.size());
		std::vector<int> rightX(rightLaneIdx.size());
		std::vector<int> rightY(rightLaneIdx.size());
		for (auto& p : leftLaneIdx)
		{
			int index = &p - &(*std::begin(leftLaneIdx));
			leftX[index] = p.x;
			leftY[index] = p.y;
		}
		for (auto& p : rightLaneIdx)
		{
			int index = &p - &(*std::begin(rightLaneIdx));
			rightX[index] = p.x;
			rightY[index] = p.y;
		}

		return std::make_tuple(leftX, leftY, rightX, rightY, showImg);
	}

	std::tuple<int, std::vector<float>, std::vector<float>, std::vector<cv::Point>, std::vector<cv::Point>, cv::Mat>
		fitPolynomial(const cv::Mat& binaryWarped, bool isShow)
	{
		auto[leftX, leftY, rightX, rightY, showImg] = findLanePixels(binaryWarped, isShow);

		// Calculate A,B,C for Ax*x + Bx + C = 0 (second order polynomial)
		auto leftFit = eigenOperations::polyfit(leftY, leftX, 2);
		auto rightFit = eigenOperations::polyfit(rightY, rightX, 2);

		int numPts = binaryWarped.rows;
		std::vector<int> leftFitX(numPts);
		std::vector<int> rightFitX(numPts);
		std::vector<cv::Point> left(numPts);
		std::vector<cv::Point> right(numPts);

		// TO DO: Ad an exception here. 
		if ((leftFit.size() == 3) && (rightFit.size() == 3))
		{
			for (int y = 0; y < numPts; ++y)
			{
				int xLeft = leftFit[2] * y * y + leftFit[1] * y + leftFit[0];
				int xRight = rightFit[2] * y * y + rightFit[1] * y + rightFit[0];
				leftFitX[y] = xLeft;
				rightFitX[y] = xRight;
				left[y] = cv::Point(xLeft, y);
				right[y] = cv::Point(xRight, y);
			}
		}
		else
		{
			for (int y = 0; y < numPts; ++y)
			{
				int xLeft = y * y + y;
				int xRight = y * y + y;
				leftFitX[y] = xLeft;
				rightFitX[y] = xRight;
				left[y] = cv::Point(xLeft, y);
				right[y] = cv::Point(xRight, y);
			}
		}

		if (isShow)
		{
			std::vector<cv::Point> leftLane(leftX.size());
			std::vector<cv::Point> rightLane(rightX.size());
			for (auto& p : leftLane)
			{
				int index = &p - &(*std::begin(leftLane));
				p = cv::Point(leftX[index], leftY[index]);
				showImg.at<cv::Vec3b>(leftY[index], leftX[index]) = { 0, 0, 255 };
			}
			for (auto& p : rightLane)
			{
				int index = &p - &(*std::begin(rightLane));
				p = cv::Point(rightX[index], rightY[index]);
				showImg.at<cv::Vec3b>(rightY[index], rightX[index]) = { 255, 0, 0 };
			}
			std::vector<std::vector<cv::Point>> leftPolyDraw{ left };
			std::vector<std::vector<cv::Point>> rightPolyDraw{ right };
			cv::polylines(showImg, leftPolyDraw, false, { 0, 255, 255 }, 2);
			cv::polylines(showImg, rightPolyDraw, false, { 0, 255, 255 }, 2);
		}

		return std::make_tuple(numPts, leftFit, rightFit, left, right, showImg);
	}

	/////////////////////////////////
	// Perspective rectangle block //
	/////////////////////////////////
	namespace
	{
		// Line segment intersection using vectors //
		cv::Point2f perp(const cv::Point2f a)
		{
			return cv::Point2f(-a.y, a.x);
		}

		// Line segment a given by endpoints a1, a2 //
		// Line segment b given by endpoints b1, b2 //
		cv::Point2f segIntersect(const cv::Point2f a1, const cv::Point2f a2, const cv::Point2f b1, const cv::Point2f b2)
		{
			auto da = a2 - a1;
			auto db = b2 - b1;
			auto dp = a1 - b1;
			auto dap = perp(da);
			auto denom = dap.x * db.x + dap.y * db.y;
			auto num = dap.x * dp.x + dap.y * dp.y;
			return num / denom * db + b1;
		}

		// Find k, b for x = ky + b //
		std::vector<float> fitLine(const std::vector<int>& xValues, const std::vector<int>& yValues)
		{
			return eigenOperations::polyfit(xValues, yValues, 1);
		}
	}

	// Convert polynomial dots to original perspective //
	std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> getDotsPair(
		const std::vector<cv::Point>& leftX, const std::vector<cv::Point>& rightX, const cv::Mat& debugImg)
	{
		int height = debugImg.rows;
		assert(leftX.size() == rightX.size());
		std::vector<cv::Point2f> left(leftX.size());
		std::vector<cv::Point2f> right(rightX.size());
		for (size_t i = 0; i < leftX.size(); ++i)
		{
			left[i] = cv::Point2f(leftX[i].x, i);
			right[i] = cv::Point2f(rightX[i].x, i);
		}
		std::vector<cv::Point2f> leftDots;
		std::vector<cv::Point2f> rightDots;
		cv::perspectiveTransform(left, leftDots, transforms::getDefaultGM(debugImg).second);
		cv::perspectiveTransform(right, rightDots, transforms::getDefaultGM(debugImg).second);
		return std::make_pair(leftDots, rightDots);
	}

	// Find perspective linear lines //
	std::pair<std::pair<float, float>, std::pair<float, float>> findLinearLines(
		const std::vector<cv::Point2f>& leftDots, const std::vector<cv::Point2f>& rightDots, 
		cv::Mat& dbgImg, bool isShow)
	{
		assert(leftDots.size() == rightDots.size());
		int height = dbgImg.rows;
		int width = dbgImg.cols;
		const float rectTopCf = 0.2361;
		int selectedSize = rectTopCf * height;
		std::vector<int> selectedLeft;
		std::vector<int> selectedRight;
		selectedLeft.reserve(selectedSize);
		selectedRight.reserve(selectedSize);
		for (auto& p : leftDots)
		{
			if ((p.x >= 0) & (p.x < width) & (p.y >= (height - selectedSize)))
			{
				int index = &p - &(*std::begin(leftDots));
				selectedLeft.emplace_back(index);
			}
		}
		for (auto& p : rightDots)
		{
			if ((p.x >= 0) & (p.x < width) & (p.y >= (height - selectedSize)))
			{
				int index = &p - &(*std::begin(rightDots));
				selectedRight.emplace_back(index);
			}
		}
		std::vector<int> leftXSel(selectedLeft.size());
		std::vector<int> leftYSel(selectedLeft.size());
		std::vector<int> rightXSel(selectedRight.size());
		std::vector<int> rightYSel(selectedRight.size());
		for (size_t i = 0; i < selectedLeft.size(); ++i)
		{
			leftXSel[i] = leftDots[selectedLeft[i]].x;
			leftYSel[i] = leftDots[selectedLeft[i]].y;
		}
		for (size_t i = 0; i < selectedRight.size(); ++i)
		{
			rightXSel[i] = rightDots[selectedRight[i]].x;
			rightYSel[i] = rightDots[selectedRight[i]].y;
		}

		auto kbLeft = fitLine(leftYSel, leftXSel);
		auto kbRight = fitLine(rightYSel, rightXSel);

		if (isShow)
		{
			assert(dbgImg.channels() == 3);
			int yTop = 0, yBot = height;
			cv::line(dbgImg, cv::Point(kbLeft[1] * yBot + kbLeft[0], yBot), cv::Point(kbLeft[1] * yTop + kbLeft[0], yTop), { 255, 255, 0 }, 2);
			cv::line(dbgImg, cv::Point(kbRight[1] * yBot + kbRight[0], yBot), cv::Point(kbRight[1] * yTop + kbRight[0], yTop), { 255, 255, 0 }, 2);
		}

		return std::make_pair(std::make_pair(kbLeft[1], kbLeft[0]), std::make_pair(kbRight[1], kbRight[0]));
	}

	std::vector<cv::Point> findPerspectiveRect(const std::pair<float, float> leftLineCoeffs,
		std::pair<float, float> rightLineCoeffs, int rectTopWidth, cv::Mat& dbgImg, bool isShow)
	{
		int height = dbgImg.rows;
		int width = dbgImg.cols;
		// find cross point 
		int yTop = 0, yBot = height;
		auto[lk, lb] = leftLineCoeffs;
		auto[rk, rb] = rightLineCoeffs;
		auto lp1 = cv::Point(lk * yBot + lb, yBot);
		auto lp2 = cv::Point(lk * yTop + lb, yTop);
		auto rp1 = cv::Point(rk * yBot + rb, yBot);
		auto rp2 = cv::Point(rk * yTop + rb, yTop);
		auto cp = segIntersect(lp1, lp2, rp1, rp2);
		auto rectBotWidth = rp1.x - lp1.x;
		auto H = yBot - cp.y;
		auto L = cp.x - lp1.x;
		auto R = rp1.x - cp.x;

		std::vector<cv::Point> rect;
		if ((cp.x >= 0) && (cp.x <= width) && (lp1.x >= 0) && (lp1.x <= width)
			&& (rp1.x >= 0) && (rp1.x <= width) && (rectBotWidth > 0) && (rectTopWidth < rectBotWidth))
		{
			auto rectHeight = H * rectTopWidth / rectBotWidth;
			rect = { lp1,
			cv::Point(cp.x - L * rectTopWidth / rectBotWidth, cp.y + rectHeight),
			cv::Point(cp.x + R * rectTopWidth / rectBotWidth, cp.y + rectHeight),
			rp1 };

			if (isShow)
			{
				std::vector<std::vector<cv::Point>> drawRect{ rect };
				cv::polylines(dbgImg, drawRect, true, { 0, 0, 255 }, 2);
			}
			return rect;
		}
		return std::vector<cv::Point>();
	}

	////////////////////////////
	// Find around poly block //
	////////////////////////////
	std::pair<std::vector<float>, std::vector<float>> fitPoly(const std::vector<int>& leftX,
		const std::vector<int>& leftY, const std::vector<int>& rightX, const std::vector<int>& rightY)
	{
		auto leftFit = eigenOperations::polyfit(leftY, leftX, 2);
		auto rightFit = eigenOperations::polyfit(rightY, rightX, 2);
		return std::make_pair(leftFit, rightFit);
	}
	// Calculate poly values depends on leftfit and rightfit coefficients //
	std::tuple<std::vector<int>, std::vector<int>, float> findPolyValues(const std::vector<float>& leftFit,
		const std::vector<float>& rightFit, cv::Size imageSize)
	{
		int vecSize = imageSize.height;
		std::vector<int> leftFitX(vecSize);
		std::vector<int> rightFitX(vecSize);
		for (int i = 0; i < vecSize; ++i)
		{
			leftFitX[i] = (leftFit[2] * i * i + leftFit[1] * i + leftFit[0]);
			rightFitX[i] = (rightFit[2] * i * i + rightFit[1] * i + rightFit[0]);
		}
		return std::make_tuple(leftFitX, rightFitX, vecSize);
	}

	// TO DO: Add Visualization if else //
	std::tuple<int, std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>, cv::Mat>
		findAroundPoly(const cv::Mat& binaryWarped, const std::vector<float>& leftFit, const std::vector<float>& rightFit)
	{
		int width = binaryWarped.cols;
		int height = binaryWarped.rows;
		// Choose the width of the margin around the previous polynomial to search //
		const float marginCf = 0.05469f;
		int margin = width * marginCf; // 70 for 1280x720

		// Find nonzero pixels
		cv::Mat nonzero;
		cv::findNonZero(binaryWarped, nonzero);
		std::vector<std::pair<int, int>> nonzeroXY(nonzero.total());
		for (size_t i = 0; i < nonzero.total(); ++i)
		{
			nonzeroXY[i] = std::make_pair(nonzero.at<cv::Point>(i).x, nonzero.at<cv::Point>(i).y);
		}

		std::vector<int> leftLaneInds;
		std::vector<int> rightLaneInds;
		leftLaneInds.reserve(nonzero.total()); // check this
		rightLaneInds.reserve(nonzero.total());
		std::vector<cv::Point> leftLaneIdx;
		std::vector<cv::Point> rightLaneIdx;
		leftLaneIdx.reserve(nonzero.total());
		rightLaneIdx.reserve(nonzero.total());
		for (auto& p : nonzeroXY)
		{
			if ((p.first > (leftFit[2] * p.second * p.second + leftFit[1] * p.second + leftFit[0] - margin)) &
				(p.first < (leftFit[2] * p.second * p.second + leftFit[1] * p.second + leftFit[0] + margin)))
			{
				leftLaneInds.emplace_back(&p - &(*std::begin(nonzeroXY)));
				leftLaneIdx.push_back(cv::Point(p.first, p.second));
			}
			if ((p.first > (rightFit[2] * p.second * p.second + rightFit[1] * p.second + rightFit[0] - margin)) &
				(p.first < (rightFit[2] * p.second * p.second + rightFit[1] * p.second + rightFit[0] + margin)))
			{
				rightLaneInds.emplace_back(&p - &(*std::begin(nonzeroXY)));
				rightLaneIdx.push_back(cv::Point(p.first, p.second));
			}
		}

		std::vector<int> leftX(leftLaneIdx.size());
		std::vector<int> leftY(leftLaneIdx.size());
		std::vector<int> rightX(rightLaneIdx.size());
		std::vector<int> rightY(rightLaneIdx.size());
		for (auto& p : leftLaneIdx)
		{
			int index = &p - &(*std::begin(leftLaneIdx));
			leftX[index] = p.x;
			leftY[index] = p.y;
		}
		for (auto& p : rightLaneIdx)
		{
			int index = &p - &(*std::begin(rightLaneIdx));
			rightX[index] = p.x;
			rightY[index] = p.y;
		}

		cv::Mat outImg;
		if (binaryWarped.channels() == 1) {
			std::vector<cv::Mat> binaryWarpedVec{ binaryWarped, binaryWarped, binaryWarped };
			cv::merge(binaryWarpedVec, outImg);
		}
		else {
			outImg = binaryWarped.clone();
		}

		int minFound = 100; // TO DO: make non constant variable (depends on image size)
		std::vector<float> newLeftFit;
		std::vector<float> newRightFit;
		newLeftFit.reserve(leftFit.size());
		newRightFit.reserve(rightFit.size());
		if ((leftX.size() < minFound) | (rightX.size() < minFound))
		{
			newLeftFit = leftFit;
			newRightFit = rightFit;
			return std::make_tuple(height, newLeftFit, newRightFit, std::vector<int>() /*None*/, std::vector<int>()/*None*/, outImg); // TO DO: add std::optional
		}

		auto newFits = fitPoly(leftX, leftY, rightX, rightY);
		newLeftFit = newFits.first;
		newRightFit = newFits.second;

		int countLeft = 0;
		for (auto& p : leftY)
		{
			if (p < (height / 2))
			{
				countLeft++;
			}
		}
		if (countLeft <= 10)
		{
			auto linCoeffs = eigenOperations::polyfit(leftY, leftX, 1);
			newLeftFit[0] = linCoeffs[0];
			newLeftFit[1] = linCoeffs[1];
			newLeftFit[2] = 0;
		}
		int countRight = 0;
		for (auto& p : rightY)
		{
			if (p < (height / 2))
			{
				countRight++;
			}
		}
		if (countRight <= 10)
		{
			auto linCoeffs = eigenOperations::polyfit(leftY, leftX, 1);
			newRightFit[0] = linCoeffs[0];
			newRightFit[1] = linCoeffs[1];
			newRightFit[2] = 0;
		}

		auto[leftFitX, rightFitX, ploty1] = findPolyValues(leftFit, rightFit, binaryWarped.size());

		auto[newLeftFitX, newRightFitX, ploty2] = findPolyValues(newLeftFit, newRightFit, binaryWarped.size());

		// Visualization
		cv::Mat windowImg = cv::Mat3b(outImg.size(), cv::Vec3b(0, 0, 0));
		std::vector<cv::Point> leftLine(height);
		std::vector<cv::Point> leftLineWindow1(height); 
		std::vector<cv::Point> leftLineWindow2(height);
		for (auto& p : leftLine)
		{
			int index = &p - &(*std::begin(leftLine));
			p = cv::Point(leftFitX[index], index);
		}
		for (auto& p : leftLineWindow1)
		{
			int index = &p - &(*std::begin(leftLineWindow1));
			p = cv::Point(leftFitX[index] - margin, index);
		}
		for (auto& p : leftLineWindow2)
		{
			int index = &p - &(*std::begin(leftLineWindow2));
			p = cv::Point(leftFitX[index] + margin, index);
		}

		std::vector<cv::Point> leftLinePts;
		leftLinePts.reserve(leftLineWindow1.size() + leftLineWindow2.size());
		// CHECK THIS
		std::reverse(leftLineWindow2.begin(), leftLineWindow2.end());
		leftLinePts.insert(leftLinePts.end(), leftLineWindow1.begin(), leftLineWindow1.end());
		leftLinePts.insert(leftLinePts.end(), leftLineWindow2.begin(), leftLineWindow2.end());
		
		std::vector<cv::Point> rightLine(height);
		std::vector<cv::Point> rightLineWindow1(height); 
		std::vector<cv::Point> rightLineWindow2(height);
		for (auto& p : rightLine)
		{
			int index = &p - &(*std::begin(rightLine));
			p = cv::Point(rightFitX[index], index);
		}
		for (auto& p : rightLineWindow1)
		{
			int index = &p - &(*std::begin(rightLineWindow1));
			p = cv::Point(rightFitX[index] - margin, index);
		}
		for (auto& p : rightLineWindow2)
		{
			int index = &p - &(*std::begin(rightLineWindow2));
			p = cv::Point(rightFitX[index] + margin, index);
		}
		std::vector<cv::Point> rightLinePts;
		rightLinePts.reserve(rightLineWindow1.size() + rightLineWindow2.size());
		std::reverse(rightLineWindow2.begin(), rightLineWindow2.end());
		rightLinePts.insert(rightLinePts.end(), rightLineWindow1.begin(), rightLineWindow1.end());
		rightLinePts.insert(rightLinePts.end(), rightLineWindow2.begin(), rightLineWindow2.end());

		// Needed for drawing
		std::vector<std::vector<cv::Point>> ll{ leftLine };
		std::vector<std::vector<cv::Point>> rr{ rightLine };
		std::vector<std::vector<cv::Point>> l{ leftLinePts };
		std::vector<std::vector<cv::Point>> r{ rightLinePts };
		cv::fillPoly(windowImg, l, { 0, 255, 0 });
		cv::fillPoly(windowImg, r, { 0, 255, 0 });
		cv::polylines(outImg, ll, false, { 0, 255, 255 }, 2);
		cv::polylines(outImg, rr, false, { 0, 255, 255 }, 2);

		cv::Mat result;
		cv::addWeighted(outImg, 1, windowImg, 0.3, 0, result);

		return std::make_tuple(height, newLeftFit, newRightFit, newLeftFitX, newRightFitX, result);
	}

	cv::Mat drawLane(const cv::Mat& image, const cv::Mat& warped, const std::vector<int>& lfx, const std::vector<int>& rfx, const cv::Mat& mInv)
	{
		cv::Mat coloredWarp(warped.size(), CV_8UC3, cv::Vec3b(0,0,0)); 

		std::vector<cv::Point> ptsLeft(warped.rows);
		std::vector<cv::Point> ptsRight(warped.rows);
		for (auto& p : ptsLeft)
		{
			int index = &p - &(*std::begin(ptsLeft));
			p = cv::Point(lfx[index], index);
		}
		for (auto& p : ptsRight)
		{
			int index = &p - &(*std::begin(ptsRight));
			p = cv::Point(rfx[index], index);
		}
		std::reverse(ptsRight.begin(), ptsRight.end());
		std::vector<cv::Point> pts;
		pts.reserve(ptsLeft.size() + ptsRight.size());
		pts.insert(pts.end(), ptsLeft.begin(), ptsLeft.end());
		pts.insert(pts.end(), ptsRight.begin(), ptsRight.end());

		std::vector<std::vector<cv::Point>> ptsFill{ pts };
		cv::fillPoly(coloredWarp, ptsFill, { 0,255,0 });

		cv::Mat unWarp;
		cv::warpPerspective(coloredWarp, unWarp, mInv, image.size());

		cv::Mat result(image.size(), image.type(), cv::Vec3b(0, 0, 0));
		cv::addWeighted(image, 1, unWarp, 0.3, 0, result);
		return result;
	}

}