#pragma once
#include "../Utils/ImageReformer.h"
#include "../Utils/LaneHandle.h"
#include "../Utils/Metric.h"

// TO DO: edit vector to list in recentXFitted, recentFit;
namespace roadLineDetection {
	struct Line
	{
		bool isDetected{ false }; // was detected in the last iteration?
		std::deque<std::vector<int>> recentXFitted; // x values of the last n fits of the line
		std::deque<std::vector<float>> recentFit; //polynomial coefficients of the last n fits of the line
		std::vector<int> bestX; //average x values of the fitted line over the last n iterations
		std::vector<float> bestFit; // polynomial coefficients averaged over the last n iterations
		std::vector<float> currentFit; // polynomial coefficinets for the most recent fit
		double rad{ 0 }; // radius of curvature of the line in some units
		std::vector<int> allX; // x values for detected line pixels
		std::vector<int> allY; // y values for detected line pixels
	};

	class Processor final
	{
	public:
		// Hyper Params. You can "play" with them.
		const int warpedPixelsThresh{ 110000 }; // HARD
		const int maxLostFrames{ 5 };
		const int historySize{ 6 };
		const int minFramesToAverage{ 3 };
		const int perspRectTopMin{ /*100*/ 50 };
		const int perspRectTopMax{ /*350*/350 }; //150 project
		const int perspRectTopDef{ /*250*/250 }; //100 project
		// Default corners for 1280 x 720. TO DO: edit this for any image scale
		std::vector<cv::Point2f> defaultSrcCorners{ cv::Point2f(220, 720), cv::Point2f(515, 500), cv::Point2f(765, 500), cv::Point2f(1060, 720) };
		std::vector<cv::Point2f> defaultDstCorners{ cv::Point2f(280, 720), cv::Point2f(280, 0), cv::Point2f(1000, 0), cv::Point2f(1000, 720) };
	public:
		Processor(bool doAverage=false, bool doDebug=false, std::vector<cv::Point2f> defCorners = std::vector<cv::Point2f>(), int defWidth = 0);

	public:
		// View range adoptive help functions //
		void DecreaseViewRange();
		void IncreaseViewRange();
		void ResetViewRange();

		std::tuple<std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>> FitDots(
			const std::vector<cv::Point2f>& leftDots, const std::vector<cv::Point2f>& rightDots) const;
		std::tuple<std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>> RefitLane(
			const cv::Mat& oldMInv, const cv::Mat& newM, const std::vector<int>& lfx, const std::vector<int>& rfx) const;
		void UpdateHistory(const cv::Mat& oldMInv, const cv::Mat& newM);
		std::tuple<bool, std::vector<float>, std::vector<float>, std::vector<int>, std::vector<int>> UpdatePerspective(
			std::vector<float>& lf, std::vector<float>& rf, std::vector<int>& lfx, std::vector<int>& rfx);              // CHECK THIS

		// Check lane properties //
		bool IsGoodLane(const std::vector<float>& leftFit, const std::vector<float>& rightFit, 
			const std::vector<int>& leftFitX, const std::vector<int>& rightFitX) const;
		bool IsSharpTurn(const std::vector<float>& lf, const std::vector<float>& rf,
			const std::vector<int>& lfx, const std::vector<int>& rfx, const int threshold) const;
		bool IsFlatLane(const std::vector<float>& lf, const std::vector<float>& rf) const;

		// Reset history
		void ResetHistory();

		bool ProcessGoodLane();
		void ProcessBadLane();

		cv::Mat DetectLane(const cv::Mat& warpedBinImg);
		
		cv::Mat DrawTextInfo(cv::Mat& img);

		// Main inference method
		cv::Mat Inference(const cv::Mat& img, bool isDeb = false);

	private:
		Line m_lLane;
		Line m_rLane;
		int m_frame;
		int m_lostFrames;
		double m_lanePos;
		double m_curv;
		int m_perspRectTop;
		int m_historyLen;
		int m_warpedPixels;
		bool m_doAverage;
		bool m_doDebug;
		std::vector<cv::Point2f> m_defCorners;
		cv::Mat m_defM;
		cv::Mat m_defMInv;

		std::vector<cv::Point2f> m_corners;
		cv::Mat m_m;
		cv::Mat m_mInv;
		cv::Mat m_visBin;

		int m_count = 0;
	};
}