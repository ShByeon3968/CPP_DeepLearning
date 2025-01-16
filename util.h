#pragma once
#include <opencv2/opencv.hpp>


// asArray �Լ�: ��Į�� ���� cv::Mat���� ��ȯ�ϰų�, cv::Mat �״�� ��ȯ
inline cv::Mat asArray(const cv::Mat& x) {
    // �Է��� �̹� cv::Mat�̶�� �״�� ��ȯ
    return x;
}

inline cv::Mat asArray(double scalar) {
    // ��Į�� ���� 1x1 ���(cv::Mat)�� ��ȯ
    return cv::Mat(1, 1, CV_64F, scalar);
}

// �Լ� �����ε��� ���� ��Į��� cv::Mat�� �����Ͽ� ó��
template <typename T>
inline cv::Mat asArray(T x) {
    static_assert(std::is_arithmetic<T>::value, "Input must be a numeric type or cv::Mat");
    // ���� Ÿ���� 1x1 ��ķ� ��ȯ
    return cv::Mat(1, 1, CV_64F, static_cast<double>(x));
}