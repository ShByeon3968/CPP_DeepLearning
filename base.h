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

private:
    void checkDataType(const cv::Mat& data)
    {
        // ������ Ÿ�� Ȯ��
        if (data.empty()) {
            throw std::invalid_argument("Input data is empty.");
        }

        // OpenCV�� Mat Ÿ�Ը� ���
        if (data.type() != CV_32F && data.type() != CV_64F) {
            throw std::invalid_argument("Input data must be of type CV_32F or CV_64F.");
        }
    }
};

class Function : public std::enable_shared_from_this<Function> {
public:
    std::vector<std::shared_ptr<Variable>> inputs;
    std::vector<std::shared_ptr<Variable>> outputs;

public:
    Function() = default;
    virtual ~Function() = default;

    // ���� �Է� �� ��� ����
    std::vector<std::shared_ptr<Variable>> operator()(const std::vector<std::shared_ptr<Variable>>& inputVars) {
        if (inputVars.empty()) {
            throw std::invalid_argument("No input variables provided.");
        }

        // �Է� ������ ����
        std::vector<cv::Mat> xs;
        for (const auto& var : inputVars) {
            if (!var) {
                throw std::invalid_argument("Null variable in inputs.");
            }
            xs.push_back(var->data);
        }

        // Forward ���� ����
        std::vector<cv::Mat> ys = forward(xs);

        // ��� ���� ����
        std::vector<std::shared_ptr<Variable>> outputVars;
        for (const auto& y : ys) {
            auto outputVar = std::make_shared<Variable>(y);
            outputVar->SetCreator(shared_from_this()); // â���� ����
            outputVars.push_back(outputVar);
        }

        // �Է� �� ��� ����
        inputs = inputVars;
        outputs = outputVars;

        return outputVars;
    }

    // Forward �� Backward �Լ�
    virtual std::vector<cv::Mat> forward(const std::vector<cv::Mat>& xs) = 0;
    virtual std::vector<cv::Mat> backward(const std::vector<cv::Mat>& gys) = 0;
};

class Square : public Function {
public:
    // Forward ����
    std::vector<cv::Mat> forward(const std::vector<cv::Mat>& xs) override {
        std::vector<cv::Mat> ys;
        for (const auto& x : xs) {
            if (x.empty()) {
                throw std::invalid_argument("Input matrix is empty.");
            }
            cv::Mat y;
            cv::pow(x, 2, y); // ���� ����
            ys.push_back(y);
        }
        return ys;
    }

    // Backward ����
    std::vector<cv::Mat> backward(const std::vector<cv::Mat>& gys) override {
        if (inputs.empty()) {
            throw std::logic_error("Inputs are not set.");
        }

        std::vector<cv::Mat> grads;
        for (size_t i = 0; i < gys.size(); ++i) {
            const cv::Mat& x = inputs[i]->data;
            const cv::Mat& gy = gys[i];
            grads.push_back(2 * x.mul(gy)); // 2 * x * gy
        }
        return grads;
    }
};

class Exp : public Function {
public:
    std::vector<cv::Mat> forward(const std::vector<cv::Mat>& xs) override {
        std::vector<cv::Mat> ys;
        for (const auto& x : xs) {
            if (x.empty()) {
                throw std::invalid_argument("Input matrix is empty.");
            }
            if (x.type() != CV_32F && x.type() != CV_64F) {
                throw std::invalid_argument("Input matrix must be of type CV_32F or CV_64F.");
            }
            cv::Mat y;
            cv::exp(x, y); // ���� ����
            ys.push_back(y);
        }
        return ys;
    }

    std::vector<cv::Mat> backward(const std::vector<cv::Mat>& gys) override {
        if (inputs.empty()) {
            throw std::logic_error("Inputs are not set.");
        }

        std::vector<cv::Mat> grads;
        for (size_t i = 0; i < gys.size(); ++i) {
            const cv::Mat& x = inputs[i]->data;
            const cv::Mat& gy = gys[i];

            cv::Mat expX;
            cv::exp(x, expX); // Forward���� ����� exp(x)�� �ٽ� ���
            grads.push_back(expX.mul(gy)); // exp(x) * gy
        }
        return grads;
    }
};

// Square ���� �Լ�
std::vector<std::shared_ptr<Variable>> square(const std::vector<std::shared_ptr<Variable>>& xs);
std::vector<std::shared_ptr<Variable>> exp(const std::vector<std::shared_ptr<Variable>>& xs);
