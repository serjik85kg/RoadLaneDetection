#include "Utils/ImageReformer.h"
#include "Utils/EigenOperations.h"
#include "Processor/Processor.h"

#include "Drawdebug/Debugdraw.h"

int main(int argc, char * argv)
{

	std::vector<cv::Point2f> defaultSrcCorners{ cv::Point2f(220, 720), cv::Point2f(580, 460), cv::Point2f(700, 460), cv::Point2f(1120, 720) }; // for test video scales

	auto processImg = roadLineDetection::Processor(true, false, defaultSrcCorners, 100); //250 harder

	cv::VideoCapture cap("data/video/challenge_video.mp4");
	while (cap.isOpened())
	{
		cv::Mat frame;
		cap.read(frame);

		if (frame.empty())
			break;

		auto result = processImg.Inference(frame, true);
		cv::imshow("result", result);
		cv::waitKey(30);
	} 


	//debugdraw::showImageReformerBinary(img);
	//debugdraw::showWarpedFitPolynomial(img);
	//debugdraw::showPerspectiveLinearLines(img);
	//debugdraw::showPerspectiveRectangle(img);
	//debugdraw::showWarpedAroundPolyArea(img);
	//debugdraw::showFinalSingleResult(img);
	system("pause");
}