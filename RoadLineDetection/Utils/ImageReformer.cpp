#include "ImageReformer.h"

namespace imageReformer {

	namespace {

		cv::Mat grayscale(const cv::Mat& image)
		{
			cv::Mat grayImage;
			cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
			return grayImage;
		}

		std::vector<cv::Mat> splitMatChannels(const cv::Mat& image)
		{
			std::vector<cv::Mat> threeChannels;
			cv::split(image, threeChannels);
			return threeChannels;
		}

		cv::Mat grayscaleSpecial(const cv::Mat& image)
		{
			auto channelsBGR = splitMatChannels(image);
			cv::Mat1b grayImageSpecial = channelsBGR[0] * 0.299 + channelsBGR[1] * 0.587 + channelsBGR[2] * 0.117;
			return grayImageSpecial;
		}

		cv::Mat convertToHSL(const cv::Mat& image)
		{
			cv::Mat hlsImage;
			cv::cvtColor(image, hlsImage, cv::COLOR_BGR2HLS);
			return hlsImage;
		}

		// Color threshold func for single channel
		cv::Mat channelThresh(const cv::Mat& channel, int threshMin, int threshMax)
		{
			cv::Mat threshedChannel;
			cv::inRange(channel, threshMin, threshMax, threshedChannel);
			return threshedChannel;
		}

		// Special scaled sobel filter
		cv::Mat sobelThreshScaled(const cv::Mat& channelFrame, int threshMin, int threshMax) {
			cv::Mat sobelImg;
			cv::Sobel(channelFrame, sobelImg, CV_64F, 1, 0);
			cv::Mat sobelImgAbs = cv::abs(sobelImg);
			double minVal, maxVal;
			cv::minMaxLoc(sobelImgAbs, &minVal, &maxVal);
			cv::Mat scaledMat = sobelImgAbs / maxVal;
			scaledMat = scaledMat * 255;
			cv::Mat sobelThr;
			cv::threshold(scaledMat, sobelThr, threshMin, threshMax, cv::THRESH_BINARY);
			sobelThr.convertTo(sobelThr, CV_8U, 1, 0);
			return sobelThr;
		}

		// CLAHE normalization
		cv::Mat clache(const cv::Mat& channel)
		{
			cv::Ptr<cv::CLAHE> clache = cv::createCLAHE(2.0, cv::Size(8, 8));
			cv::Mat normal;
			clache->apply(channel, normal);
			return normal;
		}

		// Eroder and dilater
		cv::Mat eroder(const cv::Mat& channel)
		{
			cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
			cv::Mat channelEroded;
			cv::erode(channel, channelEroded, erodeElement, cv::Point(-1, -1), 2);
			return channelEroded;
		}

		cv::Mat dilater(const cv::Mat& channel)
		{
			cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
			cv::Mat channelDilated;
			cv::dilate(channel, channelDilated, dilateElement, cv::Point(-1, -1), 1);
			return channelDilated;
		}
	}

	// Main function. Make binary final image
	std::vector<cv::Mat> makeBinary(const cv::Mat& image) // TO DO: edit output format
	{
		// Preprocessing //
		auto channelsBGR = splitMatChannels(image);
		auto channelR = channelsBGR[0];
		auto channelG = channelsBGR[1];
		auto channelB = channelsBGR[2];
		auto diffRG = cv::Mat(image.rows, image.cols, (uchar)0);
		auto diffGB = cv::Mat(image.rows, image.cols, (uchar)0);
		auto diffRB = cv::Mat(image.rows, image.cols, (uchar)0);

		cv::absdiff(channelR, channelG, diffRG);
		cv::absdiff(channelG, channelB, diffGB);
		cv::absdiff(channelR, channelB, diffRB);

		auto grayImage = grayscale(image);
		auto grayImageSpecial = grayscaleSpecial(image);

		auto hlsImage = convertToHSL(image);
		auto channelsHLS = splitMatChannels(hlsImage);
		auto channelS = channelsHLS[2];
		auto grayImageClache = clache(grayImage);
		auto channelBClache = clache(channelB);
		auto channelGClache = clache(channelG);
		auto channelRClache = clache(channelR);

		// Gray mask //
		auto grayRegRG = cv::Mat1b(grayImage.rows, grayImage.cols, (uchar)0);
		auto grayRegGB = cv::Mat1b(grayImage.rows, grayImage.cols, (uchar)0);
		auto grayRegRB = cv::Mat1b(grayImage.rows, grayImage.cols, (uchar)0);
		cv::inRange(diffRG, 0, 25, grayRegRG);
		cv::inRange(diffGB, 0, 25, grayRegGB);
		cv::inRange(diffRB, 0, 25, grayRegRB);
		cv::Mat grayReg = (grayRegRG & grayRegGB & grayRegRB);
		auto grayRegEr = eroder(grayReg);

		// S-channel filter //
		auto filterS = sobelThreshScaled(channelS, 30, 255);
		filterS -= grayRegEr;

		///////////////////
		// SOBEL FILTERS //
		///////////////////
		// Gray with removed white mask //
		auto whiteMask = channelThresh(grayImage, 180, 255);
		whiteMask = dilater(whiteMask);
		auto grayNoWhite = grayRegEr.clone();
		grayNoWhite -= whiteMask;

		// Sobel common filter //
		auto sobelX = sobelThreshScaled(grayImageClache, 30, 255);
		sobelX -= grayNoWhite;

		// Shadow filter //
		auto sobelXLow = sobelThreshScaled(grayImageClache, 15, 255);
		auto brightMask = channelThresh(grayImageSpecial, 100, 255);
		auto brightMaskDilated = dilater(brightMask);
		auto sobelShadows = sobelXLow.clone();
		sobelShadows -= brightMaskDilated;

		// Combine common and shadow filter //
		cv::Mat sobelFilter = sobelX | sobelShadows; //+

		//////////////////
		// COLOR FILTER //
		//////////////////
		// Color mask //
		auto colorRegRG = cv::Mat1b(grayImage.rows, grayImage.cols, (uchar)0);
		auto colorRegGB = cv::Mat1b(grayImage.rows, grayImage.cols, (uchar)0);
		auto colorRegRB = cv::Mat1b(grayImage.rows, grayImage.cols, (uchar)0);
		cv::threshold(diffRG, colorRegRG, 15, 255, cv::THRESH_BINARY);
		cv::threshold(diffGB, colorRegGB, 15, 255, cv::THRESH_BINARY);
		cv::threshold(diffRB, colorRegRB, 15, 255, cv::THRESH_BINARY);
		cv::Mat colorReg = colorRegRG & colorRegGB & colorRegRB; // +
		auto colorRegEroded = eroder(colorReg);

		// Yellow filter //
		cv::Mat imageClache = cv::Mat3b(image.rows, image.cols, cv::Vec3b(0, 0, 0));
		std::vector<cv::Mat> imageClacheChannels;
		imageClacheChannels.push_back(channelBClache);
		imageClacheChannels.push_back(channelGClache);
		imageClacheChannels.push_back(channelRClache);
		cv::merge(imageClacheChannels, imageClache);
		cv::Mat yellow;
		cv::inRange(imageClache, cv::Scalar(0, 160, 160), cv::Scalar(130, 255, 255), yellow); // TO DO make defines

		// White filter //
		auto white = channelThresh(grayImageClache, 210, 255);
		auto lowLightReg = channelThresh(grayImageSpecial, 0, 150);
		auto lowLightRegDilated = dilater(lowLightReg);
		white &= lowLightRegDilated;
		lowLightRegDilated.release();

		// Light filter! REMOVE asphalt border CHECK CHECK CHECK 
		auto sobelLight = sobelThreshScaled(grayImageSpecial, 25, 255);
		auto lowerLightReg = channelThresh(grayImageSpecial, 0, 120);
		auto lowerLightRegDilated = dilater(lowerLightReg);
		sobelLight -= lowerLightRegDilated;
		lowerLightRegDilated.release();

		auto colorFilter = yellow | white | sobelLight;
		colorFilter = colorFilter - colorRegEroded;

		//Output // TO DO: edit output format
		cv::Mat colorBinary = cv::Mat3b(grayImage.rows, grayImage.cols, cv::Vec3b(0, 0, 0));
		std::vector<cv::Mat> colorBinaryChannels;
		colorBinaryChannels.push_back(colorFilter);
		colorBinaryChannels.push_back(sobelFilter);
		colorBinaryChannels.push_back(filterS);
		cv::merge(colorBinaryChannels, colorBinary);

		cv::Mat combinedBinary = cv::Mat1b(colorBinary.rows, colorBinary.cols);
		combinedBinary = colorFilter | sobelFilter | filterS;

		std::vector<cv::Mat> binary;
		binary.push_back(combinedBinary);
		binary.push_back(colorBinary);
		binary.push_back(sobelFilter);

		return binary;
	}
}