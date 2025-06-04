#include "..\headers\Dense_Layer.hpp"

DenseLayer::DenseLayer(int input_dim, int output_dim, ActivationType activation, const string& name) : Layer(name, true), input_features(input_dim), output_features(output_dim) {
    
    if (input_dim <= 0 || output_dim <= 0) {
        throw invalid_argument("Input and output dimensions must be positive");
    }
    
    weight_initialization = Initializer_Weights_Biases::get_recommended(activation);

    weights = Tensor(1, 1, input_features, output_features);
    biases = Tensor(1, 1, output_features, 1);
    weight_gradients = Tensor(1, 1, input_features, output_features);
    bias_gradients = Tensor(1, 1, output_features, 1);
    
    initialize_weights();
}

DenseLayer::DenseLayer(int input_dim, int output_dim, InitializationType weight_init, const string& name) : Layer(name, true), input_features(input_dim), output_features(output_dim), weight_initialization(weight_init) {
    
    if (input_dim <= 0 || output_dim <= 0) {
        throw invalid_argument("Input and output dimensions must be positive");
    }
    
    weights = Tensor(1, 1, input_features, output_features);
    biases = Tensor(1, 1, output_features, 1);
    weight_gradients = Tensor(1, 1, input_features, output_features);
    bias_gradients = Tensor(1, 1, output_features, 1);
    
    initialize_weights();
}

DenseLayer::~DenseLayer() {}

void DenseLayer::initialize_weights() {
    Initializer_Weights_Biases::initialize(weights, weight_initialization, input_features, output_features);

    Initializer_Weights_Biases::initialize(biases, InitializationType::ZEROS, output_features);
}

void DenseLayer::add_biases_to_output() {
    double* output_data = this->output.get_data();
    const double* bias_data = biases.get_data();
    
    int total_elements = this->output.get_size();
    
    for (int i = 0; i < total_elements; i++) {
        int feature_idx = i % output_features;
        output_data[i] += bias_data[feature_idx];
    }
}

Tensor DenseLayer::forward(const Tensor& input) {
    this->input = input;
    
    if (input.get_dim1() != input_features) {
        throw invalid_argument("Input dimension mismatch. Expected " + to_string(input_features) + ", got " + to_string(input.get_dim1()));
    }
    
    this->output = input.matmult_broadcast(weights);
    
    add_biases_to_output();
    
    return this->output;
}

Tensor DenseLayer::backward(const Tensor& gradient) {
    if (gradient.get_dim1() != output_features) {
        throw invalid_argument("Gradient dimension mismatch");
    }
    
    weight_gradients.zero();
    bias_gradients.zero();

    const double* input_data = input.get_data();
    const double* grad_data = gradient.get_data();
    double* weight_grad_data = weight_gradients.get_data();
    double* bias_grad_data = bias_gradients.get_data();
    
    int batch_size = input.get_dim4();
    int channels = input.get_dim3();
    int height = input.get_dim2();

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int input_base = ((b * channels + c) * height + h) * input_features;
                int grad_base = ((b * channels + c) * height + h) * output_features;
                
                for (int i = 0; i < input_features; i++) {
                    for (int j = 0; j < output_features; j++) {
                        weight_grad_data[i * output_features + j] += 
                            input_data[input_base + i] * grad_data[grad_base + j];
                    }
                }
            }
        }
    }

    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                int grad_base = ((b * channels + c) * height + h) * output_features;
                for (int j = 0; j < output_features; j++) {
                    bias_grad_data[j] += grad_data[grad_base + j];
                }
            }
        }
    }

    Tensor weights_T = weights.transpose(2, 3);
    Tensor input_gradient = gradient.matmult_broadcast(weights_T);
    
    return input_gradient;
}

/* Tensor DenseLayer::backward(const Tensor& gradient) {
    if (gradient.get_dim1() != output_features) {
        throw invalid_argument("Gradient dimension mismatch");
    }
    
    for (int i = 0; i < weight_gradients.get_size(); i++) {
        weight_gradients.get_data()[i] = 0.0;
    }
    for (int i = 0; i < bias_gradients.get_size(); i++) {
        bias_gradients.get_data()[i] = 0.0;
    }
    
    for (int b = 0; b < input.get_dim4(); b++) {
        for (int c = 0; c < input.get_dim3(); c++) {
            for (int h = 0; h < input.get_dim2(); h++) {
                for (int i = 0; i < input_features; i++) {
                    for (int j = 0; j < output_features; j++) {
                        weight_gradients(0, 0, i, j) += input(b, c, h, i) * gradient(b, c, h, j);
                    }
                }
            }
        }
    }
    
    for (int b = 0; b < gradient.get_dim4(); b++) {
        for (int c = 0; c < gradient.get_dim3(); c++) {
            for (int h = 0; h < gradient.get_dim2(); h++) {
                for (int j = 0; j < output_features; j++) {
                    bias_gradients(0, 0, j, 0) += gradient(b, c, h, j);
                }
            }
        }
    }
  
    Tensor weights_T = weights.transpose(2, 3);
    Tensor input_gradient = gradient.matmult_broadcast(weights_T);
    
    return input_gradient;
} */

void DenseLayer::update_weights(double learning_rate) {
    weights -= weight_gradients * learning_rate;
    biases -= bias_gradients * learning_rate;
}

vector<int> DenseLayer::output_shape(const vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        throw invalid_argument("Input shape must have 4 dimensions");
    }
    
    // Dense layer only changes the last dimension
    return {input_shape[0], input_shape[1], input_shape[2], output_features};
}

int DenseLayer::param_count() const {
    return (input_features * output_features) + output_features;
}

const Tensor& DenseLayer::get_weights() const {
    return weights;
}

const Tensor& DenseLayer::get_biases() const {
    return biases;
}

void DenseLayer::set_weights(const Tensor& new_weights) {
    if (new_weights.get_dim3() != input_features || new_weights.get_dim1() != output_features) {
        throw invalid_argument("Weight dimensions don't match layer configuration. " "Expected (1, 1, " + to_string(input_features) + ", " + to_string(output_features) + "), got (1, 1, " + to_string(new_weights.get_dim3()) + ", " + to_string(new_weights.get_dim1()) + ")");
    }
    weights = new_weights;
}

void DenseLayer::set_biases(const Tensor& new_biases) {
    if (new_biases.get_dim3() != output_features || new_biases.get_dim1() != 1) {
        throw invalid_argument("Bias dimensions don't match layer configuration. " "Expected (1, 1, " + to_string(output_features) + ", 1), got (1, 1, " + to_string(new_biases.get_dim3()) + ", " + to_string(new_biases.get_dim1()) + ")");
    }
    biases = new_biases;
}