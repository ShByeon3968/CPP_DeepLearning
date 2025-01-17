#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <stdexcept>

class Function;

class Variable {
public:
    Variable();
    explicit Variable(const cv::Mat& data);

    cv::Mat data;
    cv::Mat grad;

    std::shared_ptr<Function> creator;

public:
    void SetCreator(const std::shared_ptr<Function>& func);
    void backward();
};

class Function :public std::enable_shared_from_this<Function>
{
public:
	std::shared_ptr<Variable> input;
	std::shared_ptr<Variable> output;
public:
    Function() = default;
    virtual ~Function() = default;

    std::shared_ptr<Variable> operator()(const std::shared_ptr<Variable>& inputVar) {
        if (!inputVar) {
            throw std::invalid_argument("Input variable is null.");
        }

        cv::Mat x = inputVar->data;
        cv::Mat y = forward(x);

        input = inputVar; // 입력 변수 저장
        auto outputVar = std::make_shared<Variable>(y);
        outputVar->SetCreator(shared_from_this()); // 창조자 저장
        output = outputVar;

        return outputVar;
    }

    virtual cv::Mat forward(const cv::Mat& x) = 0;
    virtual cv::Mat backward(const cv::Mat& gy) = 0;
};

class Square : public Function {
public:
    cv::Mat forward(const cv::Mat& x) override {
        if (x.empty()) {
            throw std::invalid_argument("Input matrix is empty.");
        }

        cv::Mat output;
        cv::pow(x, 2, output);
        return output;
    }

    cv::Mat backward(const cv::Mat& gy) override {
        if (!input) {
            throw std::logic_error("Input variable is not set.");
        }

        cv::Mat x = input->data;
        cv::Mat gx = 2 * x.mul(gy); // Element-wise multiplication
        return gx;
    }
};

class Exp : public Function {
public:
    cv::Mat forward(const cv::Mat& x) override {
        if (x.empty()) {
            throw std::invalid_argument("Input matrix is empty.");
        }

        if (x.type() != CV_32F && x.type() != CV_64F) {
            throw std::invalid_argument("Input matrix must be of type CV_32F or CV_64F.");
        }

        cv::Mat output;
        cv::exp(x, output);
        return output;
    }

    cv::Mat backward(const cv::Mat& gy) override {
        if (!input) {
            throw std::logic_error("Input variable is not set.");
        }

        cv::Mat x = input->data;
        cv::Mat expOutput;
        cv::exp(x, expOutput);
        cv::Mat y = expOutput.mul(gy); // Element-wise multiplication
        return y;
    }
};

// Square 연산 함수
std::shared_ptr<Variable> square(const std::shared_ptr<Variable>& x);
