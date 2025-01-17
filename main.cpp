#include "base.h"

int main() {
    // 데이터 생성
    cv::Mat data{ cv::Mat_<float>(1, 1) << 0.5f };

    // 연산 객체를 스마트 포인터로 생성
    auto B = std::make_shared<Exp>();
    auto C = std::make_shared<Square>();

    // 입력 변수 생성
    auto x = std::make_shared<Variable>(data);

    // 연산 수행
    auto a = square(x);
    auto b = (*B)(a);
    auto y = square(b);

    // 역전파 초기값 설정
    y->grad = cv::Mat_<float>(1, 1) << 1.0f;
    y->backward();

    // 결과 출력
    std::cout << x->grad << std::endl;

    return 0;
}