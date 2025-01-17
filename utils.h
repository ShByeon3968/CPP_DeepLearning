#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include "base.h"

//as Array: ��Į�� ���� cv::Mat���� ��ȯ�ϰų� cv::Mat �״�� ��ȯ
inline cv::Mat asArray(const cv::Mat& x) 
{
	// �Է��� �̹� cv::Mat�̶�� �״�� ��ȯ
	return x;
}

inline cv::Mat asArray(double scalar) 
{
	// double ��Į�� ���� 1x1 ��ķ� ��ȯ
	return cv::Mat(1, 1, CV_64F, scalar);
}

template <typename T>
inline cv::Mat asArray(T x) 
{
	static_assert(std::is_arithmetic<T>::value, "Input must be a numeric type or cv::Mat");
	return cv::Mat(1, 1, CV_64F, static_cast<double>(x));
}

template <typename T>
double numerical_diff(T f, Variable x, float eps = 1e-4)
{
	Variable x0 = Variable{ x.data - eps };
	Variable x1 = Variable{ x.data + eps };
	Variable y0 = f(x0);
	Variable y1 = f(x1);
	return (y1.data - y0.data) / (2 * eps)
}
