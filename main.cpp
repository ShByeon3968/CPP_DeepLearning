#include "base.h"

int main() {
	Square A;
	Exp B;
	Square C;

	cv::Mat data = cv::Mat_<float>(1, 1) << 0.5f;

	Variable x = Variable(data);
	Variable a = A(x);
	Variable b = B(a);
	Variable y = C(b);


	cv::Mat gy = cv::Mat_<float>(1, 1) << 1.0f;
	y.grad = gy;

	Function* c_func = y.creator;
	b = c_func->input;
	b.grad = c_func->backward(y.grad);

	Function* b_func = b.creator;
	a = b_func->input;
	a.grad = b_func->backward(b.grad);

	Function* a_func = a.creator;
	x = a_func->input;
	x.grad = a_func->backward(a.grad);
	std::cout << x.grad << std::endl;
	return 0;
}