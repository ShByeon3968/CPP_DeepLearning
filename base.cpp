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
        grad = cv::Mat::ones(data.size(), data.type()); // gradient 초기화
    }

    // 스택에 창조자 함수 추가
    std::vector<std::shared_ptr<Function>> funcs;
    if (creator) {
        funcs.push_back(creator);
    }

    // 함수 호출 그래프 탐색
    while (!funcs.empty()) {
        auto f = funcs.back();
        funcs.pop_back();

        if (!f) continue;

        // 모든 출력에서 그라디언트 수집
        std::vector<cv::Mat> gys;
        for (const auto& output : f->outputs) {
            if (output && !output->grad.empty()) {
                gys.push_back(output->grad);
            }
            else {
                throw std::logic_error("Output variable or its gradient is null.");
            }
        }

        // Backward 호출로 입력 그라디언트 계산
        std::vector<cv::Mat> gxs = f->backward(gys);

        if (gxs.size() != f->inputs.size()) {
            throw std::logic_error("The number of gradients does not match the number of inputs.");
        }
        // 입력 변수의 그라디언트 업데이트 및 다음 함수 추가
        for (size_t i = 0; i < f->inputs.size(); ++i) {
            auto& input = f->inputs[i];
            if (input) {
                if (input->grad.empty()) {
                    input->grad = gxs[i];
                }
                else {
                    input->grad += gxs[i]; // 기존 그라디언트에 축적
                }

                // 입력 변수에 창조자가 있으면 스택에 추가
                if (input->creator) {
                    funcs.push_back(input->creator);
                }
            }
        }
    }
}

// Square 연산 함수
std::vector<std::shared_ptr<Variable>> square(const std::vector<std::shared_ptr<Variable>>& xs) 
{
    auto f = std::make_shared<Square>(); // Square 연산 객체 생성
    return (*f)(xs); // 다중 입력에 대해 Square 연산 수행
}

std::vector<std::shared_ptr<Variable>> exp(const std::vector<std::shared_ptr<Variable>>& xs)
{
    auto f = std::make_shared<Exp>(); // Exp 연산 객체 생성
    return (*f)(xs); // 다중 입력에 대해 Exp 연산 수행
}
