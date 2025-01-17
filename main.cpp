#include "base.h"

int main() {
    // ������ ����
    cv::Mat data{ cv::Mat_<double>(1, 1) << 0.5f };

    // �Է� ���� ����
    auto x = std::make_shared<Variable>(data);
    std::vector<std::shared_ptr<Variable>> xs = { x };
    // ���� ����
    auto a = square(xs);
    auto b = exp(a);
    auto y = square(b);

    // ������
    y[0]->backward();

    // ��� ���
    std::cout << x->grad << std::endl;

    return 0;
}