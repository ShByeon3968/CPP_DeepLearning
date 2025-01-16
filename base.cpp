#include "base.h"

Variable::Variable()
{
}

Variable::Variable(cv::Mat _data)
{
	data = _data;
}

void Variable::SetCreator(Function* _func)
{
	this->creator = _func;
}


cv::Mat Function::forward(cv::Mat x)
{
	return cv::Mat();
}

cv::Mat Function::backward(cv::Mat gy)
{
	return cv::Mat();
}

cv::Mat Square::forward(cv::Mat x)
{
	if (x.empty())
	{
		throw std::invalid_argument("Input matrix is empty.");
	}

	cv::Mat output;
	cv::pow(x, 2, output);
	return output;
}

cv::Mat Square::backward(cv::Mat gy)
{
	cv::Mat x = input.data;
	cv::Mat gx = 2 * x.mul(gy); // OpenCV는 element-wise 곱셈을 위해 mul 사용
	return gx;
}

cv::Mat Exp::forward(cv::Mat x)
{
	if (x.empty())
	{
		throw std::invalid_argument("Input matrix is empty.");
	}

	// 입력이 부동소수점 타입인지 확인
	if (x.type() != CV_32F && x.type() != CV_64F) {
		throw std::invalid_argument("Input matrix must be of type CV_32F or CV_64F.");
	}

	cv::Mat output;
	cv::exp(x, output);
	return output;
}

cv::Mat Exp::backward(cv::Mat gy)
{
	cv::Mat x = input.data;
	cv::Mat expOutput;
	cv::exp(x, expOutput);
	cv::Mat y = expOutput.mul(gy); // OpenCV는 element-wise 곱셈을 위해 mul 사용
	return y;
}