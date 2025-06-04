#pragma once
#include "..\headers\Tensor.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>

using namespace std;

enum class PoolingType {
    MAX,
    MIN,
    AVERAGE
};

class PoolingLayer {
private:
    PoolingType pooling_type;
    int pool_size;
    int stride;
    string name;
    
    Tensor pool_indices;
    Tensor input;  

    int calculate_output_size(int input_size, int padding) const;
    int calculate_padding_needed(int input_size) const;
    Tensor apply_padding(const Tensor& input) const;
    
    void max_pooling_forward(const Tensor& padded_input, Tensor& output, int batch_size, int channels, int output_height, int output_width);
    void min_pooling_forward(const Tensor& padded_input, Tensor& output, int batch_size, int channels, int output_height, int output_width);
    void average_pooling_forward(const Tensor& padded_input, Tensor& output, int batch_size, int channels, int output_height, int output_width);

    void max_min_pooling_backward(Tensor& padded_input_grad, const Tensor& gradient, int batch_size, int channels, int output_height, int output_width);
    void average_pooling_backward(Tensor& padded_input_grad, const Tensor& gradient, int batch_size, int channels, int output_height, int output_width);
    
public:
    PoolingLayer(PoolingType type, int pool_sz, int stride_val = -1, const string& layer_name = "pooling");
    
    ~PoolingLayer();

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradient);
    
    vector<int> output_shape(const vector<int>& input_shape) const;

    PoolingType get_pooling_type() const;
    int get_pool_size() const;
    int get_stride() const;
    string get_name() const;
    
    void set_name(const string& layer_name);
    
    string get_pooling_name() const;
};