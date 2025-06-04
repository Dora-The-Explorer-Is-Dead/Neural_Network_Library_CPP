#pragma once
#include "..\headers\Layer.hpp"
#include "..\headers\Tensor.hpp"
#include "..\headers\Initializer_Weights_Biases.hpp"
#include <string>
#include <vector>
#include <stdexcept>
using namespace std;

class DenseLayer : public Layer {
private:
    int input_features;
    int output_features;
    Tensor weights;
    Tensor biases;

    Tensor weight_gradients;
    Tensor bias_gradients;

    InitializationType weight_initialization;

    void add_biases_to_output();

public:
    DenseLayer(int input_dim, int output_dim, ActivationType activation = ActivationType::RELU, const string &name = "dense");

    DenseLayer(int input_dim, int output_dim, InitializationType weight_init, const string &name = "dense");

    ~DenseLayer() override;

    Tensor forward(const Tensor &input) override;
    void initialize_weights() override;
    Tensor backward(const Tensor &gradient) override;
    void update_weights(double learning_rate) override;

    vector<int> output_shape(const vector<int> &input_shape) const override;

    int param_count() const override;

    const Tensor &get_weights() const;
    const Tensor &get_biases() const;

    void set_weights(const Tensor &new_weights);
    void set_biases(const Tensor &new_biases);
};
