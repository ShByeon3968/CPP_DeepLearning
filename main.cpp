#include "base.h"

int main() {
    // ������ ����
    cv::Mat data{ cv::Mat_<float>(1, 1) << 0.5f };

    // ���� ��ü�� ����Ʈ �����ͷ� ����
    auto B = std::make_shared<Exp>();
    auto C = std::make_shared<Square>();

    // �Է� ���� ����
    auto x = std::make_shared<Variable>(data);

    // ���� ����
    auto a = square(x);
    auto b = (*B)(a);
    auto y = square(b);

    // ������ �ʱⰪ ����
    y->grad = cv::Mat_<float>(1, 1) << 1.0f;
    y->backward();

    // ��� ���
    std::cout << x->grad << std::endl;

    return 0;
}