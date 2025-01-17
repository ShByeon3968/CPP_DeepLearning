#pragma once
#include "base.h"

Variable::Variable() = default;

Variable::Variable(const cv::Mat& _data) : data(_data) {}

void Variable::SetCreator(const std::shared_ptr<Function>& func) {
    this->creator = func;
}

void Variable::backward() {
    if (grad.empty()) {
        grad = cv::Mat::ones(data.size(), data.type()); // Initialize gradient with ones
    }

    std::vector<std::shared_ptr<Function>> funcs;
    funcs.push_back(creator);

    while (!funcs.empty()) {
        auto f = funcs.back();
        funcs.pop_back();

        if (!f) continue;

        auto x = f->input; // 입력 변수
        auto y = f->output; // 출력 변수

        if (!y) {
            throw std::logic_error("Output variable is null.");
        }

        cv::Mat gy = y->grad;
        x->grad = f->backward(gy);

        // If the input variable has a creator, add it to the stack
        if (x->creator) {
            funcs.push_back(x->creator);
        }
    }
}

// Square 연산 함수
std::shared_ptr<Variable> square(const std::shared_ptr<Variable>& x)
{
    auto f = std::make_shared<Square>();
    return (*f)(x);
}