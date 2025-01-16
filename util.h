#pragma once
#include <opencv2/opencv.hpp>


// asArray 함수: 스칼라 값을 cv::Mat으로 변환하거나, cv::Mat 그대로 반환
inline cv::Mat asArray(const cv::Mat& x) {
    // 입력이 이미 cv::Mat이라면 그대로 반환
    return x;
}

inline cv::Mat asArray(double scalar) {
    // 스칼라 값을 1x1 행렬(cv::Mat)로 변환
    return cv::Mat(1, 1, CV_64F, scalar);
}

// 함수 오버로딩을 통해 스칼라와 cv::Mat을 구분하여 처리
template <typename T>
inline cv::Mat asArray(T x) {
    static_assert(std::is_arithmetic<T>::value, "Input must be a numeric type or cv::Mat");
    // 숫자 타입을 1x1 행렬로 변환
    return cv::Mat(1, 1, CV_64F, static_cast<double>(x));
}