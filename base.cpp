#pragma once
#include "base.h"

Variable::Variable() = default;

Variable::Variable(const cv::Mat& _data) : data(_data) 
{
    checkDataType(_data);
}

void Variable::SetCreator(const std::shared_ptr<Function>& func) {
    this->creator = func;
}

void Variable::backward() {
    if (grad.empty()) {
        grad = cv::Mat::ones(data.size(), data.type()); // gradient �ʱ�ȭ
    }

    // ���ÿ� â���� �Լ� �߰�
    std::vector<std::shared_ptr<Function>> funcs;
    if (creator) {
        funcs.push_back(creator);
    }

    // �Լ� ȣ�� �׷��� Ž��
    while (!funcs.empty()) {
        auto f = funcs.back();
        funcs.pop_back();

        if (!f) continue;

        // ��� ��¿��� �׶���Ʈ ����
        std::vector<cv::Mat> gys;
        for (const auto& output : f->outputs) {
            if (output && !output->grad.empty()) {
                gys.push_back(output->grad);
            }
            else {
                throw std::logic_error("Output variable or its gradient is null.");
            }
        }

        // Backward ȣ��� �Է� �׶���Ʈ ���
        std::vector<cv::Mat> gxs = f->backward(gys);

        if (gxs.size() != f->inputs.size()) {
            throw std::logic_error("The number of gradients does not match the number of inputs.");
        }
        // �Է� ������ �׶���Ʈ ������Ʈ �� ���� �Լ� �߰�
        for (size_t i = 0; i < f->inputs.size(); ++i) {
            auto& input = f->inputs[i];
            if (input) {
                if (input->grad.empty()) {
                    input->grad = gxs[i];
                }
                else {
                    input->grad += gxs[i]; // ���� �׶���Ʈ�� ����
                }

                // �Է� ������ â���ڰ� ������ ���ÿ� �߰�
                if (input->creator) {
                    funcs.push_back(input->creator);
                }
            }
        }
    }
}

// Square ���� �Լ�
std::vector<std::shared_ptr<Variable>> square(const std::vector<std::shared_ptr<Variable>>& xs) 
{
    auto f = std::make_shared<Square>(); // Square ���� ��ü ����
    return (*f)(xs); // ���� �Է¿� ���� Square ���� ����
}

std::vector<std::shared_ptr<Variable>> exp(const std::vector<std::shared_ptr<Variable>>& xs)
{
    auto f = std::make_shared<Exp>(); // Exp ���� ��ü ����
    return (*f)(xs); // ���� �Է¿� ���� Exp ���� ����
}
