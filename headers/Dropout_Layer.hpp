#include "../headers/Tensor.hpp"
#include <random>
#include <string>
#include <vector>

using namespace std;

enum class DropoutRate {
    NONE = 0,        
    VERY_LIGHT = 1,  
    LIGHT = 2,       
    MODERATE = 3,  
    HEAVY = 4,      
    VERY_HEAVY = 5  
};

class DropoutLayer {
private:
    string name;
    double dropout_rate;
    Tensor mask;
    Tensor input;
    Tensor output;
    bool training_mode;
    mutable mt19937 generator;
    mutable uniform_real_distribution<double> distribution;
    
    static double enum_to_rate(DropoutRate preset);
    
public:
    DropoutLayer(DropoutRate preset, const string& layer_name = "dropout");

    DropoutLayer(double rate, const string& layer_name = "dropout");
    
    ~DropoutLayer();

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& gradient);

    vector<int> output_shape(const vector<int>& input_shape) const;

    void set_training(bool training);
    bool is_training() const;

    string get_name() const;
    void set_name(const string& layer_name);
    double get_dropout_rate() const;
    void set_dropout_rate(double rate);
    const Tensor& get_mask() const;
    const Tensor& get_output() const;
};

