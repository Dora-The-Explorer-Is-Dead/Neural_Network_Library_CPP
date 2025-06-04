#include "..\headers\Activation.hpp"

/*enum class ActivationType {
    RELU,       
    SIGMOID,    
    TANH,       
    SOFTMAX,    
    LEAKY_RELU  
};*/

Tensor Activation::forward(const Tensor& input, ActivationType type, double param) {
    switch (type) {
        case ActivationType::RELU:
            return relu(input);
        case ActivationType::SIGMOID:
            return sigmoid(input);
        case ActivationType::TANH:
            return tanh(input);
        case ActivationType::SOFTMAX:
            return softmax(input);
        case ActivationType::LEAKY_RELU:
            return leaky_relu(input, param);
        default:
            throw invalid_argument("Unknown activation type");
    }
}

Tensor Activation::backward(const Tensor& input, const Tensor& gradient, ActivationType type, double param) {
    switch (type) {
        case ActivationType::RELU:
            return relu_derivative(input, gradient);
        case ActivationType::SIGMOID:
            return sigmoid_derivative(input, gradient);
        case ActivationType::TANH:
            return tanh_derivative(input, gradient);
        case ActivationType::SOFTMAX:
            return softmax_derivative(input, gradient);
        case ActivationType::LEAKY_RELU:
            return leaky_relu_derivative(input, gradient, param);
        default:
            throw invalid_argument("Unknown activation type");
    }
}

Tensor Activation::relu(const Tensor& input) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    for (int i = 0; i < input.get_size(); i++) {
        output.get_data()[i] = max(0.0, input.get_data()[i]);
    }

    return output;
}

Tensor Activation::relu_derivative(const Tensor& input, const Tensor& gradient) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());

    for (int i = 0; i < input.get_size(); i++) {
        double derivative = (input.get_data()[i] > 0.0) ? 1.0 : 0.0;
        output.get_data()[i] = gradient.get_data()[i] * derivative;
    }

    return output;
}

Tensor Activation::sigmoid(const Tensor& input) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());

    for (int i = 0; i < input.get_size(); i++) {
        double x = input.get_data()[i];
        x = max(-500.0, min(500.0, x)); // clipping to prevent overflow
        output.get_data()[i] = 1.0 / (1.0 + exp(-x));
    }
    
    return output;
}

Tensor Activation::sigmoid_derivative(const Tensor& input, const Tensor& gradient) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    Tensor sigmoid_output = sigmoid(input);

    for (int i = 0; i < input.get_size(); i++) {
        double x = sigmoid_output.get_data()[i];
        double derivative = x * (1.0 - x);
        output.get_data()[i] = gradient.get_data()[i] * derivative;
    }
    
    return output;
}

Tensor Activation::tanh(const Tensor& input) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    for (int i = 0; i < input.get_size(); i++) {
        double x = input.get_data()[i];
        x = max(-500.0, min(500.0, x)); // clipping to prevent overflow
        output.get_data()[i] = std::tanh(x);
    }
    
    return output;
}

Tensor Activation::tanh_derivative(const Tensor& input, const Tensor& gradient) {
    Tensor tanh_output = tanh(input);
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    for (int i = 0; i < input.get_size(); i++) {
        double tanh_val = tanh_output.get_data()[i];
        double derivative = 1.0 - tanh_val * tanh_val;
        output.get_data()[i] = gradient.get_data()[i] * derivative;
    }
    
    return output;
}

/* Tensor Activation::softmax(const Tensor& input) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());

    for (int i = 0; i < input.get_dim4(); i++) {
        for (int j = 0; j < input.get_dim3(); j++) {
            for (int k = 0; k < input.get_dim2(); k++) {
                
                double max_val = input(i, j, k, 0);
                for (int l = 1; l < input.get_dim1(); l++) {
                    max_val = max(max_val, input(i, j, k, l));
                }

                double sum = 0.0;
                for (int l = 0; l < input.get_dim1(); l++) {
                    double exp_val = exp(input(i, j, k, l) - max_val);
                    output(i, j, k, l) = exp_val;
                    sum += exp_val;
                }
                
                for (int l = 0; l < input.get_dim1(); l++) {
                    output(i, j, k, l) /= sum;
                }
            }
        }
    }
    
    return output;
}
*/

Tensor Activation::softmax(const Tensor& input) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    const double* input_data = input.get_data();
    double* output_data = output.get_data();
    
    for (int i = 0; i < input.get_dim4(); i++) {
        for (int j = 0; j < input.get_dim3(); j++) {
            for (int k = 0; k < input.get_dim2(); k++) {
                int base_idx = ((i * input.get_dim3() + j) * input.get_dim2() + k) * input.get_dim1();
                
                double max_val = input_data[base_idx];
                for (int l = 1; l < input.get_dim1(); l++) {
                    max_val = max(max_val, input_data[base_idx + l]);
                }
                
                double sum = 0.0;
                for (int l = 0; l < input.get_dim1(); l++) {
                    double exp_val = exp(input_data[base_idx + l] - max_val);
                    output_data[base_idx + l] = exp_val;
                    sum += exp_val;
                }
                
                for (int l = 0; l < input.get_dim1(); l++) {
                    output_data[base_idx + l] /= sum;
                }
            }
        }
    }
    
    return output;
}

Tensor Activation::softmax_derivative(const Tensor& input, const Tensor& gradient) {
    Tensor softmax_output = softmax(input);
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    for (int i = 0; i < input.get_dim4(); i++) {
        for (int j = 0; j < input.get_dim3(); j++) {
            for (int k = 0; k < input.get_dim2(); k++) {
                for (int l = 0; l < input.get_dim1(); l++) {
                    double sum = 0.0;
                    
                    // Jacobian
                    for (int m = 0; m < input.get_dim1(); m++) {
                        double jacobian;
                        if (l == m) {
                            jacobian = softmax_output(i, j, k, l) * (1.0 - softmax_output(i, j, k, l));
                        } else {
                            jacobian = -softmax_output(i, j, k, l) * softmax_output(i, j, k, m);
                        }
                        sum += gradient(i, j, k, m) * jacobian;
                    }
                    
                    output(i, j, k, l) = sum;
                }
            }
        }
    }
    
    return output;
}

Tensor Activation::leaky_relu(const Tensor& input, double alpha) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    for (int i = 0; i < input.get_size(); i++) {
        double x = input.get_data()[i];
        output.get_data()[i] = (x > 0.0) ? x : alpha * x;
    }
    
    return output;
}

Tensor Activation::leaky_relu_derivative(const Tensor& input, const Tensor& gradient, double alpha) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    for (int i = 0; i < input.get_size(); i++) {
        double derivative = (input.get_data()[i] > 0.0) ? 1.0 : alpha;
        output.get_data()[i] = gradient.get_data()[i] * derivative;
    }
    
    return output;
}