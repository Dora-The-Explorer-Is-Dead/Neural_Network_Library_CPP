#pragma once
#include "..\headers\Tensor.hpp"
#include <stdexcept>
#include <cmath>
using namespace std;

enum class ActivationType {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX,
    LEAKY_RELU
};

class Activation {
public:
    static Tensor forward(const Tensor &input, ActivationType type, double param = 0.01);
    static Tensor backward(const Tensor &input, const Tensor &gradient, ActivationType type, double param = 0.01);

    static Tensor relu(const Tensor &input);
    static Tensor relu_derivative(const Tensor &input, const Tensor &gradient);

    static Tensor sigmoid(const Tensor &input);
    static Tensor sigmoid_derivative(const Tensor &input, const Tensor &gradient);

    static Tensor tanh(const Tensor &input);
    static Tensor tanh_derivative(const Tensor &input, const Tensor &gradient);

    static Tensor softmax(const Tensor &input);
    static Tensor softmax_derivative(const Tensor &input, const Tensor &gradient);

    static Tensor leaky_relu(const Tensor &input, double alpha = 0.01);
    static Tensor leaky_relu_derivative(const Tensor &input, const Tensor &gradient, double alpha = 0.01);
};
