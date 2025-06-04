#pragma once
#include "..\headers\Tensor.hpp"
#include <map>
#include <string>
#include <stdexcept>
#include <cmath>
using namespace std;

enum class OptimizerType {
    SGD,
    SGD_MOMENTUM,
    RMSPROP,
    ADAM
};

struct OptimizerParams {
    double learning_rate;
    double momentum;     // For SGD_MOMENTUM
    double beta1, beta2; // For ADAM
    double decay;        // For RMSPROP
    double epsilon;      // For ADAM and RMSPROP
    
    OptimizerParams(double lr = 0.001);
    
    static OptimizerParams sgd(double lr = 0.01);
    static OptimizerParams sgd_momentum(double lr = 0.01, double mom = 0.9);
    static OptimizerParams adam(double lr = 0.001, double b1 = 0.9, double b2 = 0.999);
    static OptimizerParams rmsprop(double lr = 0.001, double decay_rate = 0.9);
};

class Optimizer {
private:
    OptimizerType type;
    OptimizerParams params;
    
    static map<void*, Tensor> momentum_map;      // For SGD_MOMENTUM
    static map<void*, Tensor> m_map;             // For ADAM (first moment)
    static map<void*, Tensor> v_map;             // For ADAM (second moment) and RMSPROP
    static map<void*, int> step_count_map;       // For ADAM (time steps)
    
public:
    Optimizer(OptimizerType opt_type, const OptimizerParams& opt_params = OptimizerParams());
    
    void update(Tensor& weights, const Tensor& gradients);
    void update_bias(Tensor& bias, const Tensor& bias_gradients);
    
    void set_learning_rate(double lr);
    double get_learning_rate() const;
    string get_name() const;

private:
    void update_sgd(Tensor& weights, const Tensor& gradients);
    void update_sgd_momentum(Tensor& weights, const Tensor& gradients);
    void update_rmsprop(Tensor& weights, const Tensor& gradients);
    void update_adam(Tensor& weights, const Tensor& gradients);
};

Optimizer create_sgd(double lr = 0.01);
Optimizer create_adam(double lr = 0.001);
Optimizer create_sgd_momentum(double lr = 0.01, double momentum = 0.9);
Optimizer create_rmsprop(double lr = 0.001, double decay = 0.9);