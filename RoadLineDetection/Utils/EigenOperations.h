#pragma once
#include <Eigen/Dense>
#include <vector>

// TO DO: edit namespace name :DDD
namespace eigenOperations 
{
	// Finding polynomial coeffs. It is pretty implementation which use matrix transdormations. //
	// TO DO: make and alternative which doesn't depend on Eigen lib
	template <class T>
	std::vector<float> polyfit(const std::vector<T>& xValues, const std::vector<T>& yValues, int order)
	{
		assert(xValues.size() == yValues.size());
		int rows = xValues.size();
		int cols = order + 1;
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> a(xValues.size(), order + 1);
		for (size_t measIdx = 0; measIdx < a.rows(); ++measIdx)
		{
			const auto xDouble = static_cast<float>(xValues[measIdx]);
			for (size_t order = 0; order < a.cols(); ++order)
			{
				a(measIdx, order) = std::powf(xDouble, order);
			}
		}
		auto aT = a.transpose();
		auto aTa = aT * a;
		auto aTaInv = aTa.inverse();
		auto aInv = aTaInv * aT;

		Eigen::Matrix<float, Eigen::Dynamic, 1> y(yValues.size(), 1);
		for (size_t measIdx = 0; measIdx < y.rows(); ++measIdx)
		{
			y(measIdx, 0) = yValues[measIdx];
		}

		auto coefs = aInv * y;
		std::vector<float> output;
		output.reserve(coefs.rows());
		for (size_t i = 0; i < coefs.rows(); ++i)
		{
			output.push_back(coefs(i, 0));
		}
		return output;
	}


	// TO DO: edit calculateMean and remove eigenOperations //
	template <class T>
	std::vector<T> calculateMean(const std::deque<std::vector<T>>& input)
	{
		assert(input.size() > 0);
		int size = input.size();
		std::vector<T> averageVec(input.front().size());
		for (size_t i = 0; i < input.size(); ++i)
		{
			for (size_t j = 0; j < input[i].size(); ++j)
			{
				averageVec[j] += input[i][j];
			}
		}
		for (size_t i = 0; i < averageVec.size(); ++i)
		{
			averageVec[i] /= size;
		}
		return averageVec;
	}

}