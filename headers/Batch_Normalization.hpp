#pragma once
#include "..\headers\Layer.hpp"
#include "..\headers\Tensor.hpp"
#include "..\headers\Initializer_Weights_Biases.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
using namespace std;

enum class BatchNormType {
    DENSE,
    CONV2D
};

class BatchNormalization : public Layer {
private:
    BatchNormType norm_type;
    int num_features;
    double epsilon;
    double momentum;
    bool training_mode;

    Tensor gamma;
    Tensor beta;

    Tensor gamma_gradients;
    Tensor beta_gradients;

    Tensor running_mean;
    Tensor running_var;

    Tensor batch_mean;
    Tensor batch_var;
    Tensor normalized_input;

    void calculate_batch_stats(const Tensor &input);
    void update_running_stats();
    Tensor normalize_input(const Tensor &input);

public:
    BatchNormalization(int features, BatchNormType type = BatchNormType::DENSE, double eps = 1e-5, double mom = 0.1, const string &name = "batch_norm");

    ~BatchNormalization() override;

    Tensor forward(const Tensor &input) override;
    void initialize_weights() override;
    Tensor backward(const Tensor &gradient) override;
    void update_weights(double learning_rate) override;

    void set_training(bool training) override;

    vector<int> output_shape(const vector<int> &input_shape) const override;

    int param_count() const override;

    const Tensor &get_gamma() const;
    const Tensor &get_beta() const;
    const Tensor &get_running_mean() const;
    const Tensor &get_running_var() const;
    bool is_training() const;

    void set_gamma(const Tensor &new_gamma);
    void set_beta(const Tensor &new_beta);
};