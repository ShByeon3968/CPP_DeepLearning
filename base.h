#pragma once
#include <opencv2/opencv.hpp>

class Function;

class Variable
{
public:
	Variable();
	explicit Variable(cv::Mat data);

	cv::Mat data;
	cv::Mat grad;

	Function* creator;

public:
	void SetCreator(Function* _func);

};

class Function
{
public:
	Variable input;
	Variable output;
public:
	Function() = default;
	virtual ~Function() = default;
	Variable operator()(Variable _input)
	{
		// output의 참조자 반환
		cv::Mat x = _input.data;
		cv::Mat y = forward(x);
		input = _input; // 입력변수 보관
		Variable _output = Variable{ y }; 
		_output.SetCreator(this); // 출력 변수에 창조자 설정
		output = _output; // 출력 저장
		return _output;
	}

	virtual cv::Mat forward(cv::Mat x);
	virtual cv::Mat backward(cv::Mat gy);
};

class Square : public Function
{
public:
	cv::Mat forward(cv::Mat x) override;
	cv::Mat backward(cv::Mat gy) override;
};

class Exp : public Function 
{
public:
	cv::Mat forward(cv::Mat x) override;
	cv::Mat backward(cv::Mat gy) override;
};
