#pragma once
#include "..\headers\Tensor.hpp"
#include <string>
#include <vector>
#include <stdexcept>
using namespace std;

class FlattenLayer {
private:
    string name;
    vector<int> original_shape; 
    
public:
    FlattenLayer(const string& layer_name = "flatten");
    
    ~FlattenLayer();

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradient);
    
    vector<int> output_shape(const vector<int>& input_shape) const;
    
    string get_name() const;
    
    void set_name(const string& layer_name);
};