#include "base.h"

int main() {
	cv::Mat data{ cv::Mat_<float>(1,1) << 0.5f };

	// 연산 선언
	Square A;
	Exp B;
	Square C;

	Variable x{ data };
	
	Variable a = A(x);
	Variable b = B(a);
	Variable y = C(b);

	y.grad = cv::Mat_<float>(1, 1) << 1.0f;
	y.backward();

	return 0;
}