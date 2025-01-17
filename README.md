# C++를 이용한 딥러닝 구현

OpenCV의 cv::Mat 클래스를 사용하여 행렬 연산을 처리하는 C++ 기반의 간단한 딥러닝 프레임워크를 제공합니다. 이 코드는 자동 미분 기능을 포함한 간단한 계산 그래프를 구현합니다.

## 주요 기능
- **커스텀 Variable 클래스**: 행렬과 그라디언트를 지원하는 래퍼 클래스.
- **커스텀 Function 클래스**: `Square`, `Exp`와 같은 연산을 정의하기 위한 기본 클래스.
- **자동 미분**: 역전파를 통해 그라디언트를 자동으로 계산.
- **확장성**: `Function` 클래스를 상속하여 새로운 연산을 쉽게 추가할 수 있음.

## 요구 사항
- C++
- OpenCV (OpenCV 4.10.3에서 테스트됨)

## 설치 방법
1. 저장소를 클론합니다:
   ```bash
   git clone https://github.com/ShByeon3968/deep-learning-cpp.git
   cd deep-learning-cpp
   ```
2. OpenCV가 시스템에 설치되어 있는지 확인합니다.
   - Ubuntu의 경우:
     ```bash
     sudo apt update
     sudo apt install libopencv-dev
     ```
   - Windows의 경우: vcpkg를 통해 OpenCV를 설치하거나 [OpenCV 웹사이트](https://opencv.org/)에서 미리 빌드된 바이너리를 다운로드하세요.
3. 코드를 컴파일합니다:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

## 사용법
### 예제 코드
다음 예제는 프레임워크를 사용하여 그라디언트를 계산하는 방법을 보여줍니다:

```cpp
#include "base.h"
#include <iostream>

int main() {
    // 입력 데이터
    cv::Mat data{ cv::Mat_<float>(1, 1) << 0.5f };
    auto x = std::make_shared<Variable>(data);

    // 연산 수행
    auto a = square(x); // Square 연산
    auto b = exp(a);    // Exp 연산
    auto y = square(b); // Square 연산

    // 출력값에 대한 그라디언트 설정
    y->grad = cv::Mat_<float>(1, 1) << 1.0f;
    y->backward();

    // 입력 변수 x의 그라디언트 출력
    std::cout << "x.grad: " << x->grad << std::endl;

    return 0;
}
```

### 출력
위 코드에서 프로그램은 `y = ((x^2)^e)^2`에 대해 `x`에 대한 그라디언트를 계산합니다.

## 코드 구조
- `base.h`: `Variable` 및 `Function` 클래스와 `Square`, `Exp` 연산의 구현.
- `main.cpp`: 프레임워크 사용 예제.

## 새로운 함수 추가 방법
1. `Function` 클래스를 상속받아 새로운 클래스를 생성합니다.
2. `forward`와 `backward` 메서드를 구현합니다.
3. 해당 연산을 위한 래퍼 함수를 정의합니다.

예제:
```cpp
class Log : public Function {
public:
    cv::Mat forward(const cv::Mat& x) override {
        cv::Mat output;
        cv::log(x, output);
        return output;
    }

    cv::Mat backward(const cv::Mat& gy) override {
        return gy / input->data;
    }
};

std::shared_ptr<Variable> log(const std::shared_ptr<Variable>& x) {
    auto f = std::make_shared<Log>();
    return (*f)(x);
}
```
