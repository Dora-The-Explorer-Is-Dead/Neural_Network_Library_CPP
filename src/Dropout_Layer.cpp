#include "..\headers\Dropout_Layer.hpp"

double DropoutLayer::enum_to_rate(DropoutRate preset) {
    switch(preset) {
        case DropoutRate::NONE: return 0.0;
        case DropoutRate::VERY_LIGHT: return 0.05;
        case DropoutRate::LIGHT: return 0.1;
        case DropoutRate::MODERATE: return 0.2;
        case DropoutRate::HEAVY: return 0.3;
        case DropoutRate::VERY_HEAVY: return 0.5;
        default: throw invalid_argument("Unknown dropout rate preset");
    }
}

DropoutLayer::DropoutLayer(DropoutRate preset, const string& layer_name) : name(layer_name), dropout_rate(enum_to_rate(preset)), training_mode(true), distribution(0.0, 1.0) {
    random_device rd;
    generator.seed(rd());
}

DropoutLayer::DropoutLayer(double rate, const string& layer_name) : name(layer_name), dropout_rate(rate), training_mode(true), distribution(0.0, 1.0) {
    
    if (rate < 0.0 || rate >= 1.0) {
        throw invalid_argument("Dropout rate must be between 0.0 and 1.0 (exclusive)");
    }

    if (rate > 0.7) {
        cout << "Warning: Dropout rate " << rate << " is very high and may hurt performance" << endl;
    }

    random_device rd;
    generator.seed(rd());
}

DropoutLayer::~DropoutLayer() {}

Tensor DropoutLayer::forward(const Tensor& input_tensor) {
    this->input = input_tensor;
    
    if (!training_mode || dropout_rate == 0.0) {
        this->output = input_tensor;
        return this->output;
    }
    
    mask = Tensor(input_tensor.get_dim4(), input_tensor.get_dim3(), input_tensor.get_dim2(), input_tensor.get_dim1());
    this->output = Tensor(input_tensor.get_dim4(), input_tensor.get_dim3(), input_tensor.get_dim2(), input_tensor.get_dim1());
    
    double* mask_data = mask.get_data();
    const double* input_data = input_tensor.get_data();
    double* output_data = this->output.get_data();
    
    int total_elements = input_tensor.get_size();
    double scale_factor = 1.0 / (1.0 - dropout_rate);
    
    for (int i = 0; i < total_elements; i++) {
        double random_val = distribution(generator);
        
        if (random_val > dropout_rate) {
            // Keep this neuron active
            mask_data[i] = scale_factor; // scaling to maintain expected value
            output_data[i] = input_data[i] * scale_factor;
        } else {
            // Drop this neuron
            mask_data[i] = 0.0;
            output_data[i] = 0.0;
        }
    }
    
    return this->output;
}

Tensor DropoutLayer::backward(const Tensor& gradient) {
    if (!training_mode || dropout_rate == 0.0) {
        return gradient;
    }

    Tensor input_gradient(gradient.get_dim4(), gradient.get_dim3(), gradient.get_dim2(), gradient.get_dim1());
    
    const double* grad_data = gradient.get_data();
    const double* mask_data = mask.get_data();
    double* input_grad_data = input_gradient.get_data();
    
    int total_elements = gradient.get_size();

    for (int i = 0; i < total_elements; i++) {
        input_grad_data[i] = grad_data[i] * mask_data[i];
    }
    
    return input_gradient;
}

vector<int> DropoutLayer::output_shape(const vector<int>& input_shape) const {
    return input_shape;
}

void DropoutLayer::set_training(bool training) {
    training_mode = training;
}

bool DropoutLayer::is_training() const {
    return training_mode;
}

string DropoutLayer::get_name() const {
    return name;
}

void DropoutLayer::set_name(const string& layer_name) {
    name = layer_name;
}

double DropoutLayer::get_dropout_rate() const {
    return dropout_rate;
}

void DropoutLayer::set_dropout_rate(double rate) {
    if (rate < 0.0 || rate >= 1.0) {
        throw invalid_argument("Dropout rate must be between 0.0 and 1.0 (exclusive)");
    }
    dropout_rate = rate;
}

const Tensor& DropoutLayer::get_mask() const {
    return mask;
}

const Tensor& DropoutLayer::get_output() const {
    return output;
}