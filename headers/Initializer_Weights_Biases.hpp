#pragma once
#include "..\headers\Tensor.hpp"
#include "..\headers\Activation.hpp"
#include <random>
using namespace std;

enum class InitializationType {
    XAVIER_NORMAL,
    XAVIER_UNIFORM,
    HE_NORMAL,
    HE_UNIFORM,
    ZEROS           
};

class Initializer_Weights_Biases {
private:
    static void xavier_normal(Tensor& tensor, int fan_in, int fan_out);
    static void xavier_uniform(Tensor& tensor, int fan_in, int fan_out);
    static void he_normal(Tensor& tensor, int fan_in);
    static void he_uniform(Tensor& tensor, int fan_in);
    static void zeros(Tensor& tensor);

public:
    static void initialize(Tensor& tensor, InitializationType type, int fan_in, int fan_out = 0);

    static InitializationType get_recommended(ActivationType activation);
};