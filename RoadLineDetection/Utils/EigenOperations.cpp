#include "EigenOperations.h"
#include <math.h>
#include <iostream>



//namespace eigenOperations
//{
//	std::vector<float> polyfit(std::vector<int>& xValues, std::vector<int>& yValues, int order)
//	{
//		assert(xValues.size() == yValues.size());
//		int rows = xValues.size();
//		int cols = order + 1;
//		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> a(xValues.size(), order+1);
//		for (size_t measIdx = 0; measIdx < a.rows(); ++measIdx)
//		{
//			const auto xDouble = static_cast<float>(xValues[measIdx]);
//			for (size_t order = 0; order < a.cols(); ++order)
//			{
//				a(measIdx, order) = std::powf(xDouble, order);
//			}
//		}
//		auto aT = a.transpose();
//		auto aTa = aT * a;
//		auto aTaInv = aTa.inverse();
//		auto aInv = aTaInv * aT;
//
//		Eigen::Matrix<float, Eigen::Dynamic, 1> y(yValues.size(), 1);
//		for (size_t measIdx = 0; measIdx < y.rows(); ++measIdx)
//		{
//			y(measIdx, 0) = yValues[measIdx];
//		}
//
//		auto coefs = aInv * y;
//		std::vector<float> output;
//		output.reserve(coefs.rows());
//		for (size_t i = 0; i < coefs.rows(); ++i)
//		{
//			output.push_back(coefs(i, 0));
//		}
//		return output;
//	
//	}
//
//	std::vector<float> polyfit(std::vector<float>& xValues, std::vector<float>& yValues, int order)
//	{
//		assert(xValues.size() == yValues.size());
//		int rows = xValues.size();
//		int cols = order + 1;
//		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> a(xValues.size(), order + 1);
//		for (size_t measIdx = 0; measIdx < a.rows(); ++measIdx)
//		{
//			const auto xDouble = static_cast<float>(xValues[measIdx]);
//			for (size_t order = 0; order < a.cols(); ++order)
//			{
//				a(measIdx, order) = std::powf(xDouble, order);
//			}
//		}
//		auto aT = a.transpose();
//		auto aTa = aT * a;
//		auto aTaInv = aTa.inverse();
//		auto aInv = aTaInv * aT;
//
//		Eigen::Matrix<float, Eigen::Dynamic, 1> y(yValues.size(), 1);
//		for (size_t measIdx = 0; measIdx < y.rows(); ++measIdx)
//		{
//			y(measIdx, 0) = yValues[measIdx];
//		}
//
//		auto coefs = aInv * y;
//		std::vector<float> output;
//		output.reserve(coefs.rows());
//		for (size_t i = 0; i < coefs.rows(); ++i)
//		{
//			output.push_back(coefs(i, 0));
//		}
//		return output;
//
//	}
//}