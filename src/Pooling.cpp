#include "..\headers\Pooling.hpp"

PoolingLayer::PoolingLayer(PoolingType type, int pool_sz, int stride_val, const string& layer_name) : pooling_type(type), pool_size(pool_sz), name(layer_name) {
    
    if (pool_sz <= 0) {
        throw invalid_argument("Pool size must be positive");
    }
    
    stride = (stride_val == -1) ? pool_size : stride_val;
    
    if (stride <= 0) {
        throw invalid_argument("Stride must be positive");
    }
}

PoolingLayer::~PoolingLayer() {}

int PoolingLayer::calculate_output_size(int input_size, int padding) const {
    return (input_size + 2 * padding - pool_size) / stride + 1;
}

int PoolingLayer::calculate_padding_needed(int input_size) const {
    int desired_output_size = (input_size + stride - 1) / stride;
    return max(((desired_output_size - 1) * stride + pool_size - input_size + 1) / 2, 0);
}

Tensor PoolingLayer::apply_padding(const Tensor& input) const {
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    
    if (padding == 0) {
        return input;
    }
    
    int batch_size = input.get_dim4();
    int channels = input.get_dim3();
    int padded_height = input_height + 2 * padding;
    int padded_width = input_width + 2 * padding;
    
    Tensor padded_input(batch_size, channels, padded_height, padded_width);

    double pad_value = 0.0;
    if (pooling_type == PoolingType::MAX) {
        pad_value = -numeric_limits<double>::infinity();
    } else if (pooling_type == PoolingType::MIN) {
        pad_value = numeric_limits<double>::infinity();
    }
    
    int total_padded_size = batch_size * channels * padded_height * padded_width;
    for (int i = 0; i < total_padded_size; i++) {
        padded_input.get_data()[i] = pad_value;
    }
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < input_height; h++) {
                const double* src_row = &input(b, c, h, 0);
                double* dst_row = &padded_input(b, c, h + padding, padding);
                std::copy(src_row, src_row + input_width, dst_row);
            }
        }
    }
    
    return padded_input;
}

void PoolingLayer::max_pooling_forward(const Tensor& padded_input, Tensor& output, int batch_size, int channels, int output_height, int output_width) {
    int total_output_elements = batch_size * channels * output_height * output_width;

    for (int idx = 0; idx < total_output_elements; idx++) {
        int b = idx / (channels * output_height * output_width);
        int remaining = idx % (channels * output_height * output_width);
        int c = remaining / (output_height * output_width);
        remaining = remaining % (output_height * output_width);
        int oh = remaining / output_width;
        int ow = remaining % output_width;
        
        double max_val = -numeric_limits<double>::infinity();
        int selected_idx = 0;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                
                double val = padded_input(b, c, ih, iw);
                if (val > max_val) {
                    max_val = val;
                    selected_idx = ph * pool_size + pw;
                }
            }
        }
        
        output.get_data()[idx] = max_val;
        pool_indices.get_data()[idx] = selected_idx;
    }
}

void PoolingLayer::min_pooling_forward(const Tensor& padded_input, Tensor& output, int batch_size, int channels, int output_height, int output_width) {
    int total_output_elements = batch_size * channels * output_height * output_width;
    
    for (int idx = 0; idx < total_output_elements; idx++) {
        int b = idx / (channels * output_height * output_width);
        int remaining = idx % (channels * output_height * output_width);
        int c = remaining / (output_height * output_width);
        remaining = remaining % (output_height * output_width);
        int oh = remaining / output_width;
        int ow = remaining % output_width;
        
        double min_val = numeric_limits<double>::infinity();
        int selected_idx = 0;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                
                double val = padded_input(b, c, ih, iw);
                if (val < min_val) {
                    min_val = val;
                    selected_idx = ph * pool_size + pw;
                }
            }
        }
        
        output.get_data()[idx] = min_val;
        pool_indices.get_data()[idx] = selected_idx;
    }
}

void PoolingLayer::average_pooling_forward(const Tensor& padded_input, Tensor& output, int batch_size, int channels, int output_height, int output_width) {
    double pool_area = pool_size * pool_size;
    int total_output_elements = batch_size * channels * output_height * output_width;
    
    for (int idx = 0; idx < total_output_elements; idx++) {
        int b = idx / (channels * output_height * output_width);
        int remaining = idx % (channels * output_height * output_width);
        int c = remaining / (output_height * output_width);
        remaining = remaining % (output_height * output_width);
        int oh = remaining / output_width;
        int ow = remaining % output_width;
        
        double sum = 0.0;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                sum += padded_input(b, c, ih, iw);
            }
        }
        
        output.get_data()[idx] = sum / pool_area;
    }
}

Tensor PoolingLayer::forward(const Tensor& input) {
    this->input = input;
    
    int batch_size = input.get_dim4();
    int channels = input.get_dim3();
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    int output_height = calculate_output_size(input_height, padding);
    int output_width = calculate_output_size(input_width, padding);
    
    if (output_height <= 0 || output_width <= 0) {
        throw invalid_argument("Invalid output dimensions. Check pool size and stride.");
    }

    Tensor padded_input = apply_padding(input);
    Tensor output(batch_size, channels, output_height, output_width);

    if (pooling_type == PoolingType::MAX || pooling_type == PoolingType::MIN) {
        pool_indices = Tensor(batch_size, channels, output_height, output_width);
    }

    switch (pooling_type) {
        case PoolingType::MAX:
            max_pooling_forward(padded_input, output, batch_size, channels, output_height, output_width);
            break;
        case PoolingType::MIN:
            min_pooling_forward(padded_input, output, batch_size, channels, output_height, output_width);
            break;
        case PoolingType::AVERAGE:
            average_pooling_forward(padded_input, output, batch_size, channels, output_height, output_width);
            break;
        default:
            throw invalid_argument("Unknown pooling type");
    }
    
    return output;
}

void PoolingLayer::max_min_pooling_backward(Tensor& padded_input_grad, const Tensor& gradient, int batch_size, int channels, int output_height, int output_width) {
    int total_output_elements = batch_size * channels * output_height * output_width;
    
    for (int idx = 0; idx < total_output_elements; idx++) {
        int b = idx / (channels * output_height * output_width);
        int remaining = idx % (channels * output_height * output_width);
        int c = remaining / (output_height * output_width);
        remaining = remaining % (output_height * output_width);
        int oh = remaining / output_width;
        int ow = remaining % output_width;
        
        double grad_val = gradient.get_data()[idx];
        
        int selected_idx = static_cast<int>(pool_indices.get_data()[idx]);
        int ph = selected_idx / pool_size;
        int pw = selected_idx % pool_size;
        
        int ih = oh * stride + ph;
        int iw = ow * stride + pw;
        
        padded_input_grad(b, c, ih, iw) += grad_val;
    }
}

void PoolingLayer::average_pooling_backward(Tensor& padded_input_grad, const Tensor& gradient, int batch_size, int channels, int output_height, int output_width) {
    double distributed_factor = 1.0 / (pool_size * pool_size);
    int total_output_elements = batch_size * channels * output_height * output_width;
    
    for (int idx = 0; idx < total_output_elements; idx++) {
        int b = idx / (channels * output_height * output_width);
        int remaining = idx % (channels * output_height * output_width);
        int c = remaining / (output_height * output_width);
        remaining = remaining % (output_height * output_width);
        int oh = remaining / output_width;
        int ow = remaining % output_width;
        
        double grad_val = gradient.get_data()[idx];
        double distributed_grad = grad_val * distributed_factor;
        
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int ih = oh * stride + ph;
                int iw = ow * stride + pw;
                padded_input_grad(b, c, ih, iw) += distributed_grad;
            }
        }
    }
}

Tensor PoolingLayer::backward(const Tensor& gradient) {
    int batch_size = input.get_dim4();
    int channels = input.get_dim3();
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    
    int output_height = gradient.get_dim2();
    int output_width = gradient.get_dim1();
    
    if (gradient.get_dim3() != channels) {
        throw invalid_argument("Gradient channel mismatch");
    }
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    
    int padded_height = input_height + 2 * padding;
    int padded_width = input_width + 2 * padding;
    Tensor padded_input_grad(batch_size, channels, padded_height, padded_width);
    padded_input_grad.zero(); 
    
    if (pooling_type == PoolingType::MAX || pooling_type == PoolingType::MIN) {
        max_min_pooling_backward(padded_input_grad, gradient, batch_size, channels, output_height, output_width);
    } else { 
        average_pooling_backward(padded_input_grad, gradient, batch_size, channels, output_height, output_width);
    }
    
    Tensor input_gradient(batch_size, channels, input_height, input_width);
    if (padding == 0) {
        input_gradient = padded_input_grad; 
    } else {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < input_height; h++) {
                    const double* src_row = &padded_input_grad(b, c, h + padding, padding);
                    double* dst_row = &input_gradient(b, c, h, 0);
                    std::copy(src_row, src_row + input_width, dst_row);
                }
            }
        }
    }
    
    return input_gradient;
}

vector<int> PoolingLayer::output_shape(const vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        throw invalid_argument("Input shape must have 4 dimensions");
    }
    
    int batch_size = input_shape[0];
    int channels = input_shape[1];
    int input_height = input_shape[2];
    int input_width = input_shape[3];
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    int output_height = calculate_output_size(input_height, padding);
    int output_width = calculate_output_size(input_width, padding);
    
    return {batch_size, channels, output_height, output_width};
}

PoolingType PoolingLayer::get_pooling_type() const {
    return pooling_type;
}

int PoolingLayer::get_pool_size() const {
    return pool_size;
}

int PoolingLayer::get_stride() const {
    return stride;
}

string PoolingLayer::get_name() const {
    return name;
}

void PoolingLayer::set_name(const string& layer_name) {
    name = layer_name;
}

string PoolingLayer::get_pooling_name() const {
    switch (pooling_type) {
        case PoolingType::MAX: return "Max Pooling";
        case PoolingType::MIN: return "Min Pooling";
        case PoolingType::AVERAGE: return "Average Pooling";
        default: return "Unknown Pooling";
    }
}