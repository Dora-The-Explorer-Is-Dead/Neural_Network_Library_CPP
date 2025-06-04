#include "..\headers\Initializer_Weights_Biases.hpp"

void Initializer_Weights_Biases::initialize(Tensor& tensor, InitializationType type, int fan_in, int fan_out) {
    switch (type) {
        case InitializationType::XAVIER_NORMAL:
            xavier_normal(tensor, fan_in, fan_out);
            break;
        case InitializationType::XAVIER_UNIFORM:
            xavier_uniform(tensor, fan_in, fan_out);
            break;
        case InitializationType::HE_NORMAL:
            he_normal(tensor, fan_in);
            break;
        case InitializationType::HE_UNIFORM:
            he_uniform(tensor, fan_in);
            break;
        case InitializationType::ZEROS:
            zeros(tensor);
            break;
        default:
            throw invalid_argument("Unknown initialization type");
    }
}

InitializationType Initializer_Weights_Biases::get_recommended(ActivationType activation) {
    switch (activation) {
        case ActivationType::RELU:
        case ActivationType::LEAKY_RELU:
            return InitializationType::HE_NORMAL;
            
        case ActivationType::SIGMOID:
        case ActivationType::TANH:
        case ActivationType::SOFTMAX:
            return InitializationType::XAVIER_NORMAL;
            
        default:
            return InitializationType::HE_NORMAL;
    }
}

void Initializer_Weights_Biases::xavier_normal(Tensor& tensor, int fan_in, int fan_out) {
    random_device rd;
    mt19937 gen(rd());
    double std_dev = sqrt(2.0 / (fan_in + fan_out));
    normal_distribution<double> dist(0.0, std_dev);
    
    for (int i = 0; i < tensor.get_size(); i++) {
        tensor.get_data()[i] = dist(gen);
    }
} 

void Initializer_Weights_Biases::xavier_uniform(Tensor& tensor, int fan_in, int fan_out) {
    random_device rd;
    mt19937 gen(rd());
    double limit = sqrt(6.0 / (fan_in + fan_out));
    uniform_real_distribution<double> dist(-limit, limit);
    
    for (int i = 0; i < tensor.get_size(); i++) {
        tensor.get_data()[i] = dist(gen);
    }
}

void Initializer_Weights_Biases::he_normal(Tensor& tensor, int fan_in) {
    random_device rd;
    mt19937 gen(rd());
    double std_dev = sqrt(2.0 / fan_in);
    normal_distribution<double> dist(0.0, std_dev);
    
    for (int i = 0; i < tensor.get_size(); i++) {
        tensor.get_data()[i] = dist(gen);
    }
}

void Initializer_Weights_Biases::he_uniform(Tensor& tensor, int fan_in) {
    random_device rd;
    mt19937 gen(rd());
    double limit = sqrt(6.0 / fan_in);
    uniform_real_distribution<double> dist(-limit, limit);
    
    for (int i = 0; i < tensor.get_size(); i++) {
        tensor.get_data()[i] = dist(gen);
    }
}

void Initializer_Weights_Biases::zeros(Tensor& tensor) {
    for (int i = 0; i < tensor.get_size(); i++) {
        tensor.get_data()[i] = 0.0;
    }
}