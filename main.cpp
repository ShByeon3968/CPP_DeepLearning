#include "base.h"

int main() {
    // 데이터 생성
    cv::Mat data{ cv::Mat_<double>(1, 1) << 0.5f };

    // 입력 변수 생성
    auto x = std::make_shared<Variable>(data);
    std::vector<std::shared_ptr<Variable>> xs = { x };
    // 연산 수행
    auto a = square(xs);
    auto b = exp(a);
    auto y = square(b);

    // 역전파
    y[0]->backward();

    // 결과 출력
    std::cout << x->grad << std::endl;

    return 0;
}