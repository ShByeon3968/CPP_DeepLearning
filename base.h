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
		// output�� ������ ��ȯ
		cv::Mat x = _input.data;
		cv::Mat y = forward(x);
		input = _input; // �Էº��� ����
		Variable _output = Variable{ y }; 
		_output.SetCreator(this); // ��� ������ â���� ����
		output = _output; // ��� ����
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
