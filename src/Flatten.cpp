#include "..\headers\Flatten.hpp"

FlattenLayer::FlattenLayer(const string& layer_name) : name(layer_name) {}

FlattenLayer::~FlattenLayer() {}

Tensor FlattenLayer::forward(const Tensor& input) {
    original_shape = {input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1()};
    
    int batch_size = input.get_dim4();
    int flattened_size = input.get_dim3() * input.get_dim2() * input.get_dim1();
    
    if (flattened_size <= 0) {
        throw invalid_argument("Invalid tensor dimensions for flattening");
    }
    
    Tensor output = input.reshape(batch_size, 1, 1, flattened_size);
    // Tensor output = input.reshape(batch_size, flattened_size, 1, 1);

    return output;
}

Tensor FlattenLayer::backward(const Tensor& gradient) {
    if (original_shape.empty()) {
        throw runtime_error("Forward pass must be called before backward pass");
    }

    Tensor input_gradient = gradient.reshape(original_shape[0], original_shape[1], original_shape[2], original_shape[3]);
    
    return input_gradient;
}

vector<int> FlattenLayer::output_shape(const vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        throw invalid_argument("Input shape must have 4 dimensions");
    }
    
    int batch_size = input_shape[0];
    int flattened_size = input_shape[1] * input_shape[2] * input_shape[3];
    
    return {batch_size, 1, 1, flattened_size};
}

string FlattenLayer::get_name() const {
    return name;
}

void FlattenLayer::set_name(const string& layer_name) {
    name = layer_name;
}