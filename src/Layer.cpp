#include "..\headers\Layer.hpp"

Layer::Layer(const string& layer_name, bool is_trainable) : name(layer_name), trainable(is_trainable) {}

Layer::~Layer() {}

void Layer::initialize_weights() {}

void Layer::set_training(bool training_mode) {
    trainable = training_mode;
}

bool Layer::is_trainable() const {
    return trainable;
}

string Layer::get_name() const {
    return name;
}

void Layer::set_name(const string& layer_name) {
    name = layer_name;
}

const Tensor& Layer::get_output() const {
    return output;
}

int Layer::param_count() const {
    return 0;
}