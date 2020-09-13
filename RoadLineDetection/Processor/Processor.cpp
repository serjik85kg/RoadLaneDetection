#include "Processor.h"
#include "../Utils/EigenOperations.h"

namespace roadLineDetection
{
	Processor::Processor(bool doAverage, bool doDebug, std::vector<cv::Point2f> defCorners, int defWidth)
		: m_frame(0),
		m_lostFrames(maxLostFrames),
		m_lanePos(0),
		m_curv(0),
		m_historyLen(0),
		m_warpedPixels(0),
		m_doAverage(doAverage),
		m_doDebug(doDebug)
	{
		if (defWidth != 0)
		{
			m_perspRectTop = defWidth;
		}
		else
		{
			m_perspRectTop = perspRectTopDef;
		}

		if (defCorners.size() == 4)
		{
			m_defCorners = defCorners;
		}
		else
		{
			m_defCorners = defaultSrcCorners;
		}
		m_lLane = Line();
		m_rLane = Line();
		auto transMat = laneHandle::transforms::calcPerspectiveMat(m_defCorners, defaultDstCorners);
		//for (auto& p : m_defCorners)
		//{
		//	std::cout << p << ' ';
		//}
		//std::cout << std::endl;
		//for (auto& p : defaultDstCorners)
		//{
		//	std::cout << p << ' ';
		//}
		//std::cout << std::endl;
		m_defM = transMat.first;
		m_defMInv = transMat.second;
		//std::cout << m_defM << std::endl;
		//std::cout << m_defMInv << std::endl;
		m_corners = m_defCorners;
		m_m = m_defM.clone();
		m_mInv = m_defMInv.clone();
		m_visBin = cv::Mat3b(cv::Size(1280, 720), cv::Vec3b(0,0,0));
	}

	void Processor::DecreaseViewRange()
	{
		m_perspRectTop += 10;
		if (m_perspRectTop > perspRectTopMax)
		{
			m_perspRectTop = perspRectTopMax;
		}
	}

	void Processor::IncreaseViewRange()
	{
		m_perspRectTop -= 10;
		if (m_perspRectTop < perspRectTopMin)
		{
			m_perspRectTop = perspRectTopMin;
		}
	}

	void Processor::ResetViewRange()
	{
		//std::cout << "ResetViewRange " << std::endl;
		m_perspRectTop = perspRectTopDef;
	}


	std::tuple<std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>> Processor::FitDots(
		const std::vector<cv::Point2f>& leftDots, const std::vector<cv::Point2f>& rightDots
	) const
	{
		std::vector<int> leftX(leftDots.size());
		std::vector<int> leftY(leftDots.size());
		std::vector<int> rightX(rightDots.size());
		std::vector<int> rightY(rightDots.size());
		for (auto& p : leftDots)
		{
			int index = &p - &(*std::begin(leftDots));
			leftX[index] = p.x;
			leftY[index] = p.y;
		}
		for (auto& p : rightDots)
		{
			int index = &p - &(*std::begin(rightDots));
			rightX[index] = p.x;
			rightY[index] = p.y;
		}
		auto[lf, rf] = laneHandle::fitPoly(leftX, leftY, rightX, rightY);
		auto[lfx, rfx, height] = laneHandle::findPolyValues(lf, rf, m_visBin.size()); //m_visBin(1280x720)
		return std::make_tuple(lf, rf, lfx, rfx);
	}

	std::tuple<std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>> Processor::RefitLane(
		const cv::Mat& oldMInv, const cv::Mat& newM, const std::vector<int>& lfx, const std::vector<int>& rfx) const
	{
		// Convert back to top view
		int height = m_visBin.rows;
		std::vector<cv::Point2f> left(lfx.size());
		std::vector<cv::Point2f> right(rfx.size());
		for (size_t i = 0; i < lfx.size(); ++i)
		{
			left[i] = cv::Point2f(lfx[i], i);
			right[i] = cv::Point2f(rfx[i], i);
		}
		std::vector<cv::Point2f> leftDots;
		leftDots.reserve(left.size());
		std::vector<cv::Point2f> rightDots;
		rightDots.reserve(right.size());
		cv::perspectiveTransform(left, leftDots, oldMInv);
		cv::perspectiveTransform(right, rightDots, oldMInv);
		// Convert to perspective view with new perspective transform
		std::vector<cv::Point2f> leftDotsP;
		leftDotsP.reserve(leftDots.size());
		std::vector<cv::Point2f> rightDotsP;
		rightDotsP.reserve(rightDots.size());
		cv::perspectiveTransform(leftDots, leftDotsP, newM);
		cv::perspectiveTransform(rightDots, rightDotsP, newM);
		return Processor::FitDots(leftDotsP, rightDotsP);
	}

	void Processor::UpdateHistory(const cv::Mat& oldMInv, const cv::Mat& newM)
	{
		for (int i = 0; i < m_historyLen; ++i)
		{
			auto[lf, rf, lfx, rfx] = Processor::RefitLane(oldMInv, newM, m_lLane.recentXFitted[i], m_rLane.recentXFitted[i]);
			m_lLane.recentFit[i] = lf;
			m_rLane.recentFit[i] = rf;
			m_lLane.recentXFitted[i] = lfx;
			m_rLane.recentXFitted[i] = rfx;
		}
	}

	std::tuple<bool, std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>> Processor::UpdatePerspective(
		std::vector<float>& lf, std::vector<float>& rf, std::vector<int>& lfx, std::vector<int>& rfx)
	{
		// translate coordinates to original image
		std::vector<cv::Point2f> left(lfx.size());
		std::vector<cv::Point2f> right(rfx.size());
		for (size_t i = 0; i < lfx.size(); ++i)
		{
			left[i] = cv::Point2f(lfx[i], i);
			right[i] = cv::Point2f(rfx[i], i);
		}
		std::vector<cv::Point2f> leftDots;
		leftDots.reserve(left.size());
		std::vector<cv::Point2f> rightDots;
		rightDots.reserve(right.size());
		cv::perspectiveTransform(left, leftDots, m_mInv);
		cv::perspectiveTransform(right, rightDots, m_mInv);

		// Find linear approximation
		auto[lLine, rLine] = laneHandle::findLinearLines(leftDots, rightDots, m_visBin, false);
		// Find new perspective matrix
		auto corners = laneHandle::findPerspectiveRect(lLine, rLine, m_perspRectTop, m_visBin, false);
		
		cv::Mat newM, newMInv;
		if (corners.size() == 4)
		{
			std::vector<cv::Point2f> corners2f(corners.size());
			//std::copy(corners.begin(), corners.end(), corners2f.begin());
			for (size_t i = 0; i < corners.size(); ++i)
			{
				corners2f[i] = corners[i];
			}
			auto transMatPairNew = laneHandle::transforms::calcPerspectiveMat(corners2f, defaultDstCorners);
			newM = transMatPairNew.first;
			newMInv = transMatPairNew.second;
		}

		if ((corners.size() == 4) && (newM.rows == 3) && (newM.cols == 3))
		{
			// Update current fit
			std::vector<cv::Point2f> leftDotsP;
			std::vector<cv::Point2f> rightDotsP;
			cv::perspectiveTransform(leftDots, leftDotsP, newM);
			cv::perspectiveTransform(rightDots, rightDotsP, newM);
			auto tuple = Processor::FitDots(leftDotsP, rightDotsP);
			lf = std::get<0>(tuple);
			rf = std::get<1>(tuple);
			lfx = std::get<2>(tuple);
			rfx = std::get<3>(tuple);

			if (!Processor::IsGoodLane(lf, rf, lfx, rfx))
			{
				return std::make_tuple(false, lf, rf, lfx, rfx);
			}

			if (Processor::m_doDebug)
			{
				std::vector<std::vector<cv::Point>> drawCorners{ corners };
				cv::polylines(m_visBin, drawCorners, true, { 0,0,255 }, 2);
			}

			Processor::UpdateHistory(m_mInv, newM);
			std::vector<cv::Point2f> cornersFloat(corners.size());
			for (size_t i = 0; i < cornersFloat.size(); ++i)
			{
				cornersFloat[i] = corners[i];
			}
			//std::copy(corners.begin(), corners.end(), cornersFloat.begin());

			m_m = newM.clone();
			m_mInv = newMInv.clone();
			m_corners = cornersFloat;	
		}

		return std::make_tuple(true, lf, rf, lfx, rfx);
	}

	bool Processor::IsGoodLane(const std::vector<float>& leftFit, const std::vector<float>& rightFit,
		const std::vector<int>& leftFitX, const std::vector<int>& rightFitX) const
	{
		if (leftFitX.size() == 0) // && (rightFitX.size() == 0))
		{
			return false;
		}
		//If lane cuvature changed too agressively, detection is incorrect
		if (Processor::m_historyLen > 0)
		{
			auto lDiff = abs(leftFit[2] - m_lLane.recentFit.back()[2]);
			auto rDiff = abs(rightFit[2] - m_rLane.recentFit.back()[2]);
			if ((lDiff > 0.0005) || (rDiff > 0.0005))
			{
				return false;
			}
		}

		// If lanes have too different curvature, detection is incorrect
		if ((std::min(abs(leftFit[2]), abs(rightFit[2])) < 0.0002) &&
			(abs(leftFit[2] - rightFit[2]) > 0.0006))
		{
			return false;
		}

		if (abs(leftFit[2] - rightFit[2]) > 0.001)
		{
			return false;
		}

		const float minDistCoeff = 0.43f;
		const float maxDistCoeff = 0.664f;
		int minDist = m_visBin.cols * minDistCoeff; // 550 for 128o width
		int maxDist = m_visBin.cols * maxDistCoeff; // 850 for 1280 width
		// If distance between lines(in the bootom of image) is too big or too small, lane was detected incorrectly //
		auto distBot = abs(leftFitX.back() - rightFitX.back());
		if ((distBot < minDist) || (distBot > maxDist))
		{
			return false;
		}
		return true;
	}

	bool Processor::IsSharpTurn(const std::vector<float>& lf, const std::vector<float>& rf,
		const std::vector<int>& lfx, const std::vector<int>& rfx, const int threshold) const
	{
		const float leftThreshCoeff = 0.04f;
		int leftThresh = m_visBin.cols * leftThreshCoeff; // 50 for 1280 width
		const float rightThreshCoeff = 0.96f;
		int rightThresh = m_visBin.cols * rightThreshCoeff; // 1230 for 1280 width
		auto foundLeft = std::find_if(lfx.begin(), lfx.end(), //np.any(lfx < leftThresh)
			[leftThresh](const auto& ele)
		{
			return (ele < leftThresh);
		});
		auto foundRight = std::find_if(rfx.begin(), rfx.end(), //np.any(rfx > rightThresh)
			[rightThresh](const auto& ele)
		{
			return (ele > rightThresh);
		});
		bool isSharp = true;
		if ((foundLeft == lfx.end()) || (foundRight == rfx.end()))
		{
			isSharp = false;
		}
		if (((abs(lf[2]) > threshold) || (abs(rf[2]) > threshold))
			|| isSharp)
		{
			return true;
		}
		return false;
	}

	bool Processor::IsFlatLane(const std::vector<float>& lf, const std::vector<float>& rf) const
	{
		if ((abs(lf[2]) < 0.0002) && (abs(rf[2]) < 0.0002))
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	void Processor::ResetHistory()
	{
		m_lLane.recentXFitted.clear();
		m_rLane.recentXFitted.clear();
		m_lLane.recentFit.clear();
		m_rLane.recentFit.clear();
		m_historyLen = 0;
	}

	bool Processor::ProcessGoodLane()
	{
		auto lf = m_lLane.currentFit;
		auto rf = m_rLane.currentFit;
		auto lfx = m_lLane.allX;
		auto rfx = m_rLane.allX;
		auto py = m_lLane.allY;

		auto isGood = true;
		if (m_lostFrames >= maxLostFrames)
		{
			Processor::ResetViewRange(); // TO DO: edit this
			auto tuple = Processor::UpdatePerspective(lf, rf, lfx, rfx);
			isGood = std::get<0>(tuple);
			lf = std::get<1>(tuple);
			rf = std::get<2>(tuple);
			lfx = std::get<3>(tuple);
			rfx = std::get<4>(tuple);
		}
		else if (Processor::IsSharpTurn(lf, rf, lfx, rfx, 0.00035))
		{
			Processor::DecreaseViewRange();
			auto tuple = Processor::UpdatePerspective(lf, rf, lfx, rfx);
			isGood = std::get<0>(tuple);
			lf = std::get<1>(tuple);
			rf = std::get<2>(tuple);
			lfx = std::get<3>(tuple);
			rfx = std::get<4>(tuple);
		}
		else if (Processor::IsFlatLane(lf, rf))
		{
			Processor::IncreaseViewRange();
			auto tuple = Processor::UpdatePerspective(lf, rf, lfx, rfx);
			isGood = std::get<0>(tuple);
			lf = std::get<1>(tuple);
			rf = std::get<2>(tuple);
			lfx = std::get<3>(tuple);
			rfx = std::get<4>(tuple);
		}
		
		m_lLane.currentFit = lf;
		m_rLane.currentFit = rf;
		m_lLane.allX = lfx;
		m_rLane.allX = rfx;

		if (!isGood)
		{
			return false;
		}

		m_lLane.isDetected = true;
		m_rLane.isDetected = true;
		m_lostFrames = 0;

		m_lLane.recentXFitted.emplace_back(lfx);
		m_rLane.recentXFitted.emplace_back(rfx);
		m_lLane.recentFit.emplace_back(lf);
		m_rLane.recentFit.emplace_back(rf);
		m_historyLen = m_lLane.recentXFitted.size();
		if (m_historyLen > historySize)
		{
			m_lLane.recentXFitted.pop_front();
			m_rLane.recentXFitted.pop_front();
			m_lLane.recentFit.pop_front();
			//std::cout << 
			m_rLane.recentFit.pop_front();
			m_historyLen--;
		}

		// Additional: Calculate the radius of curvature in pixels for both lane lines //
		auto[leftCurverad, rightCurverad, pos] = metric::measureCurvatureReal(m_visBin.size(), lfx, rfx, m_mInv);
		m_lLane.rad = leftCurverad;
		m_rLane.rad = rightCurverad;
		assert(m_historyLen == m_lLane.recentXFitted.size());
		return true;
	}

	void Processor::ProcessBadLane()
	{
		m_lLane.isDetected = false;
		m_rLane.isDetected = false;

		if (m_historyLen > 1)
		{
			m_lLane.recentXFitted.pop_front();
			m_rLane.recentXFitted.pop_front();
			m_lLane.recentFit.pop_front();
			m_rLane.recentFit.pop_front();
			m_historyLen--;
		}
		else if (m_historyLen == 0)
		{
			m_lLane.bestX.clear(); // None
			m_rLane.bestX.clear(); // None
		}

		m_lostFrames++;
		if (m_lostFrames >= maxLostFrames)
		{
			// Reset persp transform matrix
			m_m = m_defM.clone();
			m_mInv = m_defMInv.clone();
			m_corners = m_defCorners;
			m_lLane.bestX.clear(); // None
			m_rLane.bestX.clear(); // None
			Processor::ResetHistory();
		}
	}

	cv::Mat Processor::DetectLane(const cv::Mat& warpedBinImg)
	{
		if (m_lostFrames < maxLostFrames)
		{
			auto[ploty, lf, rf, lfx, rfx, visWarped] = laneHandle::findAroundPoly(warpedBinImg, m_lLane.recentFit.back(), m_rLane.recentFit.back());
			m_lLane.allX = lfx;
			m_rLane.allX = rfx;
			m_lLane.currentFit = lf;
			m_rLane.currentFit = rf;
			m_lLane.allY.resize(ploty);
			for (auto& p : m_lLane.allY)
				p = &p - &(*std::begin(m_lLane.allY));
			m_rLane.allY.resize(ploty);
			for (auto& p : m_rLane.allY)
				p = &p - &(*std::begin(m_rLane.allY));
			return visWarped;
		}
		else
		{
			auto[ploty, lf, rf, lfx, rfx, visWarped] = laneHandle::fitPolynomial(warpedBinImg, true); // to do: edit vector<type> lfx, rfx
			m_lLane.allX.resize(lfx.size());
			for (auto& p : m_lLane.allX)
			{
				int index = &p - &(*std::begin(m_lLane.allX));
				p = lfx[index].x;
			}
			m_rLane.allX.resize(rfx.size());
			for (auto& p : m_rLane.allX)
			{
				int index = &p - &(*std::begin(m_rLane.allX));
				p = rfx[index].x;
			}
			m_lLane.currentFit = lf;
			m_rLane.currentFit = rf;
			m_lLane.allY.resize(ploty);
			for (auto& p : m_lLane.allY)
				p = &p - &(*std::begin(m_lLane.allY));
			m_rLane.allY.resize(ploty);
			for (auto& p : m_rLane.allY)
				p = &p - &(*std::begin(m_rLane.allY));
			return visWarped;
		}
	}

	cv::Mat Processor::DrawTextInfo(cv::Mat& img)
	{
		auto lf = m_lLane.currentFit;
		auto rf = m_rLane.currentFit;

		cv::Scalar color;
		if (((m_doAverage) && (m_historyLen >= minFramesToAverage)) 
			|| ((!m_doAverage) && (m_historyLen >= 1)))
		{
			color = { 0, 255, 0 };
		}
		else
		{
			color = { 0, 0, 255 };
		}

		double txtSizeBig = 0.8;
		double txtSize = 0.5;
		int txtReg = 1;
		int txtBold = 2;

		auto font = cv::FONT_HERSHEY_DUPLEX;
		// TO DO: add 'None' case for rad and lane_pos
		auto rad = m_curv;
		auto pos = m_lanePos;
		//cv::putText(img, "Radius: " + std::to_string(rad) + " m", cv::Point(50, 50), font, txtSizeBig, color, txtBold, cv::LINE_AA); //deprecated
		cv::putText(img, "Position: " + std::to_string(pos) + " m", cv::Point(50, 90), font, txtSizeBig, color, txtBold, cv::LINE_AA);

		if (m_doDebug)
		{
			if (m_lLane.isDetected)
			{
				color = { 0, 255, 0 };
			}
			else
			{
				color = { 0, 0, 255 };
			}
			cv::putText(img, "Frame: " + std::to_string(m_frame),
				cv::Point(50, 140), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);
			cv::putText(img, "Found: " + std::to_string(m_lLane.isDetected && m_rLane.isDetected),
				cv::Point(200, 140), font, txtSize, color, txtReg, cv::LINE_AA);
			cv::putText(img, "History len: " + std::to_string(m_historyLen),
				cv::Point(340, 140), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);
			cv::putText(img, "Frames lost: " + std::to_string(m_lostFrames),
				cv::Point(480, 140), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);

			cv::putText(img, "View range: " + std::to_string(m_perspRectTop),
				cv::Point(50, 160), font, txtSize, { 0,255,0 }, txtReg, cv::LINE_AA);
			cv::putText(img, "Warped pix: " + std::to_string(m_warpedPixels),
				cv::Point(200, 160), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);

			if (m_lLane.allX.size() > 0)
			{
				cv::putText(img, "Width bot: " + std::to_string(abs(m_lLane.allX.back() - m_rLane.allX.back())),
					cv::Point(380, 160), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);
			}

			// TO DO: add an exception lf.size() == 0 or rf.size() == 0
			cv::putText(img, "Left fit: " + std::to_string(lf[2]) + " " + std::to_string(lf[1]) + " "
				+ std::to_string(lf[0]), cv::Point(50, 200), font, txtSize, color, txtReg, cv::LINE_AA);
			cv::putText(img, "Right fit: " + std::to_string(rf[2]) + " " + std::to_string(rf[1]) + " "
				+ std::to_string(rf[0]), cv::Point(50, 220), font, txtSize, color, txtReg, cv::LINE_AA);
			// TO DO: add substract(lf - rf)
			cv::putText(img, "Best left fit: " + std::to_string(m_lLane.bestFit[2]) + " " + std::to_string(m_lLane.bestFit[1]) + " "
				+ std::to_string(m_lLane.bestFit[0]), cv::Point(50, 280), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);
			cv::putText(img, "Best right fit: " + std::to_string(m_rLane.bestFit[2]) + " " + std::to_string(m_rLane.bestFit[1]) + " "
				+ std::to_string(m_lLane.bestFit[0]), cv::Point(50, 300), font, txtSize, { 0, 255, 0 }, txtReg, cv::LINE_AA);
		}

		return img;
	}

	cv::Mat Processor::Inference(const cv::Mat& img, bool isDeb)
	{
		// Make binary
		std::vector<cv::Mat> imgBinaryMats = imageReformer::makeBinary(img);
		auto binImg = imgBinaryMats[0];
		m_visBin = imgBinaryMats[1];

		// Draw current perspective transform rectangle
		std::vector<cv::Point> corners2i(m_corners.size());
		for (auto& p : corners2i)
		{
			int index = &p - &(*std::begin(corners2i));
			p = m_corners[index];
		}
		//std::copy(m_corners.begin(), m_corners.end(), corners2i.begin());
		std::vector<std::vector<cv::Point>> drawCorners{ corners2i };
		cv::polylines(m_visBin, drawCorners, true, { 0, 255, 255 }, 2);

		// Perspective Transform //
		auto warpedBinImg = laneHandle::transforms::perspectiveTransform(binImg, m_m);
		cv::Mat visWarped/* = cv::Mat(warpedBinImg.size(), 16, cv::Vec3b(0, 0, 0))*/;
		cv::Mat nonzero;
		cv::findNonZero(warpedBinImg, nonzero);
		m_warpedPixels = nonzero.total();
		if (m_warpedPixels < warpedPixelsThresh)
		{
			// Detect lane //
			visWarped = Processor::DetectLane(warpedBinImg);

			// Process lanes info //
			auto isGood = Processor::IsGoodLane(m_lLane.currentFit,
				m_rLane.currentFit, m_lLane.allX, m_rLane.allX);

			if (isGood)
			{
				isGood = Processor::ProcessGoodLane();
			}
			else
			{
				Processor::ProcessBadLane();
			}

			// Draw lane approximations on warped image //
			if ((isGood) && (m_lLane.allX.size() > 0))
			{
				std::vector<cv::Point> leftLaneAllXY(m_lLane.allX.size());
				for (auto& p : leftLaneAllXY)
				{
					int index = &p - &(*std::begin(leftLaneAllXY));
					p = cv::Point(m_lLane.allX[index], m_lLane.allY[index]);
				}
				std::vector<cv::Point> rightLaneAllXY(m_rLane.allX.size());
				for (auto& p : rightLaneAllXY)
				{
					int index = &p - &(*std::begin(rightLaneAllXY));
					p = cv::Point(m_rLane.allX[index], m_rLane.allY[index]);
				}
				std::vector<std::vector<cv::Point>> leftDraw{ leftLaneAllXY };
				std::vector<std::vector<cv::Point>> rightDraw{ rightLaneAllXY };
				cv::polylines(visWarped, leftDraw, false, { 0,0,255 }, 2);
				cv::polylines(visWarped, rightDraw, false, { 0,0,255 }, 2);
			}
		}
		else
		{
			std::vector<cv::Mat> warpedBinImgVec{ warpedBinImg, warpedBinImg, warpedBinImg };
			cv::merge(warpedBinImgVec, visWarped);
			Processor::ProcessBadLane();
		}

		cv::Mat result;
		// Calculate and draw approximations //
		if (((m_doAverage) && (m_historyLen >= minFramesToAverage))
			|| ((!m_doAverage) && (m_historyLen >= 1)))
		{
			//m_lLane.bestX 
			m_lLane.bestX = eigenOperations::calculateMean(m_lLane.recentXFitted);
			m_rLane.bestX = eigenOperations::calculateMean(m_rLane.recentXFitted);
			m_lLane.bestFit = eigenOperations::calculateMean(m_lLane.recentFit);
			m_rLane.bestFit = eigenOperations::calculateMean(m_rLane.recentFit);

			// Draw lane approximation //
			//cv::Mat mInv = (cv::Mat_<double>(3,3) << 1.3888e-1, -7.9e-1, 5.542295e2, 1.584476e-16, -5.17693766e-1, 4.54188568e2, 2.23704891e-19, -1.23177265e-3, 1.);
			result = laneHandle::drawLane(img, warpedBinImg, m_lLane.bestX, m_rLane.bestX, m_mInv);

			// Calculate the radius of curvature in pixels for both lane lines
			std::tuple<double, double, double> metric;
			if (m_doAverage)
			{
				metric = metric::measureCurvatureReal(img.size(), m_lLane.bestX, m_rLane.bestX, m_mInv);
			}
			else
			{
				metric = metric::measureCurvatureReal(img.size(), m_lLane.allX, m_rLane.allX, m_mInv);
			}
			m_curv = (std::get<0>(metric) + std::get<1>(metric)) / 2;
			m_lanePos = std::get<2>(metric);
		} 
		else
		{
			result = img.clone();
			m_curv = 0; // TO DO: edit to nan
			m_lanePos = 0; // TO DO: edit to nan
		}
		
		result = Processor::DrawTextInfo(result);
		m_frame++;

		if (m_doDebug)
		{
			return result;
		}
		else
		{
			return result;
		}

	}

}