#pragma once
#include "..\headers\Tensor.hpp"
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
using namespace std;

enum class LossType {
    MEAN_SQUARED_ERROR,     
    CROSS_ENTROPY,         
    BINARY_CROSS_ENTROPY,  
    MEAN_ABSOLUTE_ERROR     
};

class Loss {
public:
    static double forward(const Tensor& predictions, const Tensor& targets, LossType type);
    static Tensor backward(const Tensor& predictions, const Tensor& targets, LossType type);

    static double mean_squared_error(const Tensor& predictions, const Tensor& targets);
    static Tensor mean_squared_error_backward(const Tensor& predictions, const Tensor& targets);
    
    static double cross_entropy(const Tensor& predictions, const Tensor& targets);
    static Tensor cross_entropy_backward(const Tensor& predictions, const Tensor& targets);
    
    static double binary_cross_entropy(const Tensor& predictions, const Tensor& targets);
    static Tensor binary_cross_entropy_backward(const Tensor& predictions, const Tensor& targets);
    
    static double mean_absolute_error(const Tensor& predictions, const Tensor& targets);
    static Tensor mean_absolute_error_backward(const Tensor& predictions, const Tensor& targets);
    
    static string get_loss_name(LossType type);

private:
    static double safe_log(double x);          
    static void validate_shapes(const Tensor& predictions, const Tensor& targets);
    static int get_batch_size(const Tensor& tensor);
};