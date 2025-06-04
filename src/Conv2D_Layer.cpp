#include "..\headers\Conv2D_Layer.hpp"

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_sz, int stride_val, ActivationType activation, const string& name) : Layer(name, true), input_channels(in_channels), output_channels(out_channels), kernel_size(kernel_sz), stride(stride_val) {
    
    if (in_channels <= 0 || out_channels <= 0 || kernel_sz <= 0) {
        throw invalid_argument("Channels and kernel size must be positive");
    }
    if (stride <= 0) {
        throw invalid_argument("Stride must be positive");
    }
    
    weight_initialization = Initializer_Weights_Biases::get_recommended(activation);

    int kernels_per_filter = input_channels * kernel_size * kernel_size;
    filters = Tensor(1, 1, output_channels, kernels_per_filter);
    biases = Tensor(1, 1, output_channels, 1);
    filter_gradients = Tensor(1, 1, output_channels, kernels_per_filter);
    bias_gradients = Tensor(1, 1, output_channels, 1);
    
    initialize_weights();
}

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_sz, int stride_val, InitializationType weight_init, const string& name) : Layer(name, true), input_channels(in_channels), output_channels(out_channels), kernel_size(kernel_sz), stride(stride_val), weight_initialization(weight_init) {
    
    if (in_channels <= 0 || out_channels <= 0 || kernel_sz <= 0) {
        throw invalid_argument("Channels and kernel size must be positive");
    }
    if (stride <= 0) {
        throw invalid_argument("Stride must be positive");
    }
    
    int kernels_per_filter = input_channels * kernel_size * kernel_size;
    filters = Tensor(1, 1, output_channels, kernels_per_filter);
    biases = Tensor(1, 1, output_channels, 1);
    filter_gradients = Tensor(1, 1, output_channels, kernels_per_filter);
    bias_gradients = Tensor(1, 1, output_channels, 1);
    
    initialize_weights();
}

Conv2DLayer::~Conv2DLayer() {}

void Conv2DLayer::initialize_weights() {
    int fan_in = input_channels * kernel_size * kernel_size;
    int fan_out = output_channels * kernel_size * kernel_size;
    
    Initializer_Weights_Biases::initialize(filters, weight_initialization, fan_in, fan_out);
    Initializer_Weights_Biases::initialize(biases, InitializationType::ZEROS, output_channels);
}

int Conv2DLayer::calculate_output_size(int input_size, int padding) const {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

int Conv2DLayer::calculate_padding_needed(int input_size) const {
    int desired_output_size = (input_size + stride - 1) / stride;
    return max(((desired_output_size - 1) * stride + kernel_size - input_size + 1) / 2, 0);
}

Tensor Conv2DLayer::apply_padding(const Tensor& input) const {
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
    padded_input.zero();
    
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

/* void Conv2DLayer::im2col(const Tensor& input, Tensor& col_output) const {
    int batch_size = input.get_dim4();
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;
    int output_spatial_size = output_height * output_width;
    int kernels_per_filter = input_channels * kernel_size * kernel_size;

    col_output = Tensor(batch_size * output_spatial_size, 1, 1, kernels_per_filter);

    for (int b = 0; b < batch_size; b++) {
        for (int spatial_idx = 0; spatial_idx < output_spatial_size; spatial_idx++) {
            int oh = spatial_idx / output_width;
            int ow = spatial_idx % output_width;
            
            int patch_idx = b * output_spatial_size + spatial_idx;
            
            for (int kernel_idx = 0; kernel_idx < kernels_per_filter; kernel_idx++) {
                int ic = kernel_idx / (kernel_size * kernel_size);
                int remaining = kernel_idx % (kernel_size * kernel_size);
                int kh = remaining / kernel_size;
                int kw = remaining % kernel_size;
                
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                col_output(patch_idx, 0, 0, kernel_idx) = input(b, ic, ih, iw);
            }
        }
    }
} */

void Conv2DLayer::im2col(const Tensor& input, Tensor& col_output) const {
    int batch_size = input.get_dim4();
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;
    int output_spatial_size = output_height * output_width;
    int kernels_per_filter = input_channels * kernel_size * kernel_size;

    col_output = Tensor(batch_size * output_spatial_size, 1, 1, kernels_per_filter);

    const double* input_data = input.get_data();
    double* col_data = col_output.get_data();

    for (int b = 0; b < batch_size; b++) {
        for (int spatial_idx = 0; spatial_idx < output_spatial_size; spatial_idx++) {
            int oh = spatial_idx / output_width;
            int ow = spatial_idx % output_width;
            
            int patch_idx = b * output_spatial_size + spatial_idx;
            
            for (int kernel_idx = 0; kernel_idx < kernels_per_filter; kernel_idx++) {
                int ic = kernel_idx / (kernel_size * kernel_size);
                int remaining = kernel_idx % (kernel_size * kernel_size);
                int kh = remaining / kernel_size;
                int kw = remaining % kernel_size;
                
                int ih = oh * stride + kh;
                int iw = ow * stride + kw;

                int input_idx = ((b * input_channels + ic) * input_height + ih) * input_width + iw;
                int col_idx = patch_idx * kernels_per_filter + kernel_idx;
                
                col_data[col_idx] = input_data[input_idx];
            }
        }
    }
}

/* void Conv2DLayer::col2im(const Tensor& col_input, Tensor& im_output) const {
    int batch_size = im_output.get_dim4();
    int input_height = im_output.get_dim2();
    int input_width = im_output.get_dim1();
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    
    int padded_height = input_height + 2 * padding;
    int padded_width = input_width + 2 * padding;
    int output_height = (padded_height - kernel_size) / stride + 1;
    int output_width = (padded_width - kernel_size) / stride + 1;
    int output_spatial_size = output_height * output_width;
    int kernels_per_filter = input_channels * kernel_size * kernel_size;
    
    Tensor padded_output(batch_size, input_channels, padded_height, padded_width);
    padded_output.zero();
    
    int total_patches = batch_size * output_spatial_size;
    
    for (int patch_idx = 0; patch_idx < total_patches; patch_idx++) {
        int b = patch_idx / output_spatial_size;
        int spatial_idx = patch_idx % output_spatial_size;
        int oh = spatial_idx / output_width;
        int ow = spatial_idx % output_width;

        for (int kernel_idx = 0; kernel_idx < kernels_per_filter; kernel_idx++) {
            int ic = kernel_idx / (kernel_size * kernel_size);
            int remaining = kernel_idx % (kernel_size * kernel_size);
            int kh = remaining / kernel_size;
            int kw = remaining % kernel_size;
            
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            padded_output(b, ic, ih, iw) += col_input(patch_idx, 0, 0, kernel_idx);
        }
    }
    
    im_output.zero();
    if (padding == 0) {
        im_output = padded_output;
    } else {
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < input_height; h++) {
                    const double* src_row = &padded_output(b, c, h + padding, padding);
                    double* dst_row = &im_output(b, c, h, 0);
                    std::copy(src_row, src_row + input_width, dst_row);
                }
            }
        }
    }
}
*/

void Conv2DLayer::col2im(const Tensor& col_input, Tensor& im_output) const {
    int batch_size = im_output.get_dim4();
    int input_height = im_output.get_dim2();
    int input_width = im_output.get_dim1();
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    
    int padded_height = input_height + 2 * padding;
    int padded_width = input_width + 2 * padding;
    int output_height = (padded_height - kernel_size) / stride + 1;
    int output_width = (padded_width - kernel_size) / stride + 1;
    int output_spatial_size = output_height * output_width;
    int kernels_per_filter = input_channels * kernel_size * kernel_size;
    
    Tensor padded_output(batch_size, input_channels, padded_height, padded_width);
    padded_output.zero();

    const double* col_data = col_input.get_data();
    double* padded_data = padded_output.get_data();
    
    int total_patches = batch_size * output_spatial_size;
    
    for (int patch_idx = 0; patch_idx < total_patches; patch_idx++) {
        int b = patch_idx / output_spatial_size;
        int spatial_idx = patch_idx % output_spatial_size;
        int oh = spatial_idx / output_width;
        int ow = spatial_idx % output_width;

        for (int kernel_idx = 0; kernel_idx < kernels_per_filter; kernel_idx++) {
            int ic = kernel_idx / (kernel_size * kernel_size);
            int remaining = kernel_idx % (kernel_size * kernel_size);
            int kh = remaining / kernel_size;
            int kw = remaining % kernel_size;
            
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            int padded_idx = ((b * input_channels + ic) * padded_height + ih) * padded_width + iw;
            int col_idx = patch_idx * kernels_per_filter + kernel_idx;
            
            padded_data[padded_idx] += col_data[col_idx];
        }
    }
    
    im_output.zero();
    if (padding == 0) {
        im_output = padded_output;
    } else {
        const double* padded_src = padded_output.get_data();
        double* output_dst = im_output.get_data();
        
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < input_channels; c++) {
                for (int h = 0; h < input_height; h++) {
                    int src_offset = ((b * input_channels + c) * padded_height + (h + padding)) * padded_width + padding;
                    int dst_offset = ((b * input_channels + c) * input_height + h) * input_width;
                    
                    std::copy(padded_src + src_offset, padded_src + src_offset + input_width, output_dst + dst_offset);
                }
            }
        }
    }
}

void Conv2DLayer::add_biases_to_output() {
    int batch_size = output.get_dim4();
    int output_height = output.get_dim2();
    int output_width = output.get_dim1();
    int spatial_size = output_height * output_width;
    
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < output_channels; oc++) {
            double bias_val = biases(0, 0, oc, 0);
            
            for (int spatial_idx = 0; spatial_idx < spatial_size; spatial_idx++) {
                int oh = spatial_idx / output_width;
                int ow = spatial_idx % output_width;
                output(b, oc, oh, ow) += bias_val;
            }
        }
    }
}

Tensor Conv2DLayer::forward(const Tensor& input) {
    this->input = input;
    
    if (input.get_dim3() != input_channels) {
        throw invalid_argument("Input channel mismatch. Expected " + to_string(input_channels) + ", got " + to_string(input.get_dim3()));
    }
    
    int batch_size = input.get_dim4();
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    int output_height = calculate_output_size(input_height, padding);
    int output_width = calculate_output_size(input_width, padding);
    
    if (output_height <= 0 || output_width <= 0) {
        throw invalid_argument("Invalid output dimensions. Check kernel size and stride.");
    }
    
    Tensor padded_input = apply_padding(input);
    im2col(padded_input, col_buffer);
    Tensor filters_T = filters.transpose(2, 3);
    Tensor conv_result = col_buffer.matmult_broadcast(filters_T);
    this->output = conv_result.reshape(batch_size, output_channels, output_height, output_width);
    add_biases_to_output();
    
    return this->output;
}

Tensor Conv2DLayer::backward(const Tensor& gradient) {
    int batch_size = input.get_dim4();
    int input_height = input.get_dim2();
    int input_width = input.get_dim1();
    int output_height = gradient.get_dim2();
    int output_width = gradient.get_dim1();
    int output_spatial_size = output_height * output_width;
    
    if (gradient.get_dim3() != output_channels) {
        throw invalid_argument("Gradient channel mismatch");
    }
    
    filter_gradients.zero();
    bias_gradients.zero();
    
    Tensor padded_input = apply_padding(input);
    Tensor grad_reshaped = gradient.reshape(batch_size * output_spatial_size, 1, 1, output_channels);
    
    Tensor col_for_grad;
    im2col(padded_input, col_for_grad);
    
    Tensor grad_T = grad_reshaped.transpose(0, 3).reshape(1, 1, output_channels, batch_size * output_spatial_size);
    Tensor col_T = col_for_grad.transpose(0, 3).reshape(1, 1, input_channels * kernel_size * kernel_size, batch_size * output_spatial_size);
    
    filter_gradients = grad_T.matmult(col_T.transpose(2, 3));
    
    for (int oc = 0; oc < output_channels; oc++) {
        double sum = 0.0;
        int total_elements = batch_size * output_spatial_size;
        for (int flat_idx = 0; flat_idx < total_elements; flat_idx++) {
            int b = flat_idx / output_spatial_size;
            int spatial_idx = flat_idx % output_spatial_size;
            int oh = spatial_idx / output_width;
            int ow = spatial_idx % output_width;
            sum += gradient(b, oc, oh, ow);
        }
        bias_gradients(0, 0, oc, 0) = sum;
    }

    Tensor input_grad_col = grad_reshaped.matmult_broadcast(filters);
    Tensor input_gradient(batch_size, input_channels, input_height, input_width);
    col2im(input_grad_col, input_gradient);
    
    return input_gradient;
}

void Conv2DLayer::update_weights(double learning_rate) {
    filters -= filter_gradients * learning_rate;
    biases -= bias_gradients * learning_rate;
}

vector<int> Conv2DLayer::output_shape(const vector<int>& input_shape) const {
    if (input_shape.size() != 4) {
        throw invalid_argument("Input shape must have 4 dimensions");
    }
    
    int batch_size = input_shape[0];
    int input_height = input_shape[2];
    int input_width = input_shape[1];
    
    int padding = max(calculate_padding_needed(input_height), calculate_padding_needed(input_width));
    int output_height = calculate_output_size(input_height, padding);
    int output_width = calculate_output_size(input_width, padding);
    
    return {batch_size, output_channels, output_height, output_width};
}

int Conv2DLayer::param_count() const {
    return (output_channels * input_channels * kernel_size * kernel_size) + output_channels;
}

const Tensor& Conv2DLayer::get_filters() const {return filters;}
const Tensor& Conv2DLayer::get_biases() const {return biases;}
int Conv2DLayer::get_kernel_size() const {return kernel_size;}
int Conv2DLayer::get_stride() const {return stride;}
int Conv2DLayer::get_input_channels() const {return input_channels;}
int Conv2DLayer::get_output_channels() const {return output_channels;}

void Conv2DLayer::set_filters(const Tensor& new_filters) {
    int kernels_per_filter = input_channels * kernel_size * kernel_size;
    if (new_filters.get_dim3() != output_channels || 
        new_filters.get_dim1() != kernels_per_filter) {
        throw invalid_argument("Filter dimensions don't match layer configuration");
    }
    filters = new_filters;
}

void Conv2DLayer::set_biases(const Tensor& new_biases) {
    if (new_biases.get_dim3() != output_channels || new_biases.get_dim1() != 1) {
        throw invalid_argument("Bias dimensions don't match layer configuration");
    }
    biases = new_biases;
}