#pragma once
#include "..\headers\Layer.hpp"
#include "..\headers\Tensor.hpp"
#include "..\headers\Initializer_Weights_Biases.hpp"
#include "..\headers\Activation.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
using namespace std;

class Conv2DLayer : public Layer {
private:
    int input_channels;
    int output_channels;
    int kernel_size; // deals with square kernels only, for now
    int stride;
    
    Tensor filters; // Each row is one filter containing all its kernels flattened       
    Tensor biases;            
    Tensor filter_gradients;  
    Tensor bias_gradients;   

    Tensor col_buffer; // im2col output
    
    InitializationType weight_initialization;
    
    Tensor apply_padding(const Tensor& input) const;
    void im2col(const Tensor& input, Tensor& col_output) const;
    void col2im(const Tensor& col_input, Tensor& im_output) const;
    void add_biases_to_output();
    int calculate_output_size(int input_size, int padding) const;
    int calculate_padding_needed(int input_size) const;

public:
    Conv2DLayer(int in_channels, int out_channels, int kernel_sz, int stride_val = 1, ActivationType activation = ActivationType::RELU, const string& name = "conv2d");
    
    Conv2DLayer(int in_channels, int out_channels, int kernel_sz, int stride_val, InitializationType weight_init, const string& name = "conv2d");
    
    ~Conv2DLayer() override;

    Tensor forward(const Tensor& input) override;
    void initialize_weights() override;
    Tensor backward(const Tensor& gradient) override;
    void update_weights(double learning_rate) override;

    vector<int> output_shape(const vector<int>& input_shape) const override;
    
    int param_count() const override;
    
    const Tensor& get_filters() const;
    const Tensor& get_biases() const;
    int get_kernel_size() const;
    int get_stride() const;
    int get_input_channels() const;
    int get_output_channels() const;

    void set_filters(const Tensor& new_filters);
    void set_biases(const Tensor& new_biases);
};