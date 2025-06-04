#pragma once
#include "..\headers\Tensor.hpp"
#include <iostream>
#include <vector>
#include <string>
using namespace std;

class Layer { // Parent class for only trainable layers. The stateless layers are independent classes
protected:
    string name;
    bool trainable;
    Tensor input;  
    Tensor output; 

    vector<Tensor> weight_gradients;

public:
    Layer(const string& layer_name = "", bool is_trainable = true);
    
    virtual ~Layer();

    virtual Tensor forward(const Tensor& input) = 0;
    virtual void initialize_weights();
    virtual Tensor backward(const Tensor& gradient) = 0;
    virtual void update_weights(double learning_rate) = 0;
    
    virtual vector<int> output_shape(const vector<int>& input_shape) const = 0;

    virtual void set_training(bool training_mode);

    bool is_trainable() const;
    string get_name() const;
    void set_name(const string& layer_name);
    
    const Tensor& get_output() const;
    
    virtual int param_count() const;
};