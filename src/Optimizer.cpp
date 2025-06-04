#include "..\headers\Optimizer.hpp"

map<void*, Tensor> Optimizer::momentum_map;
map<void*, Tensor> Optimizer::m_map;
map<void*, Tensor> Optimizer::v_map;
map<void*, int> Optimizer::step_count_map;

OptimizerParams::OptimizerParams(double lr) : learning_rate(lr), momentum(0.9), beta1(0.9), beta2(0.999), decay(0.9), epsilon(1e-8) {}

OptimizerParams OptimizerParams::sgd(double lr) {
    OptimizerParams params(lr);
    return params;
}

OptimizerParams OptimizerParams::sgd_momentum(double lr, double mom) {
    OptimizerParams params(lr);
    params.momentum = mom;
    return params;
}

OptimizerParams OptimizerParams::adam(double lr, double b1, double b2) {
    OptimizerParams params(lr);
    params.beta1 = b1;
    params.beta2 = b2;
    return params;
}

OptimizerParams OptimizerParams::rmsprop(double lr, double decay_rate) {
    OptimizerParams params(lr);
    params.decay = decay_rate;
    return params;
}

Optimizer::Optimizer(OptimizerType opt_type, const OptimizerParams& opt_params) : type(opt_type), params(opt_params) {}

void Optimizer::update(Tensor& weights, const Tensor& gradients) {
    switch (type) {
        case OptimizerType::SGD:
            update_sgd(weights, gradients);
            break;
        case OptimizerType::SGD_MOMENTUM:
            update_sgd_momentum(weights, gradients);
            break;
        case OptimizerType::RMSPROP:
            update_rmsprop(weights, gradients);
            break;
        case OptimizerType::ADAM:
            update_adam(weights, gradients);
            break;
        default:
            throw invalid_argument("Unknown optimizer type");
    }
}

void Optimizer::update_bias(Tensor& bias, const Tensor& bias_gradients) {
    update(bias, bias_gradients);
}

void Optimizer::set_learning_rate(double lr) { 
    params.learning_rate = lr; 
}

double Optimizer::get_learning_rate() const { 
    return params.learning_rate; 
}

string Optimizer::get_name() const {
    switch (type) {
        case OptimizerType::SGD: return "SGD";
        case OptimizerType::SGD_MOMENTUM: return "SGD_Momentum";
        case OptimizerType::RMSPROP: return "RMSprop";
        case OptimizerType::ADAM: return "Adam";
        default: return "Unknown";
    }
}

void Optimizer::update_sgd(Tensor& weights, const Tensor& gradients) {
    weights -= gradients * params.learning_rate;
}

void Optimizer::update_sgd_momentum(Tensor& weights, const Tensor& gradients) {
    void* weights_ptr = &weights;
    
    // Initialize momentum if first time
    if (momentum_map.find(weights_ptr) == momentum_map.end()) {
        momentum_map[weights_ptr] = Tensor(weights.get_dim4(), weights.get_dim3(), weights.get_dim2(), weights.get_dim1());
        momentum_map[weights_ptr].zero();
    }
    
    Tensor& velocity = momentum_map[weights_ptr];
    
    velocity *= params.momentum;
    velocity += gradients * params.learning_rate;
    
    weights -= velocity;
}

void Optimizer::update_rmsprop(Tensor& weights, const Tensor& gradients) {
    void* weights_ptr = &weights;
    
    // Initialize squared gradient average if first time
    if (v_map.find(weights_ptr) == v_map.end()) {
        v_map[weights_ptr] = Tensor(weights.get_dim4(), weights.get_dim3(), weights.get_dim2(), weights.get_dim1());
        v_map[weights_ptr].zero();
    }
    
    Tensor& v = v_map[weights_ptr];
    
    Tensor grad_squared = gradients * gradients;
    v *= params.decay;
    v += grad_squared * (1.0 - params.decay);
    
    Tensor v_sqrt = v.sqrt();
    v_sqrt += params.epsilon;
    Tensor update = gradients / v_sqrt * params.learning_rate;
    weights -= update;
}

void Optimizer::update_adam(Tensor& weights, const Tensor& gradients) {
    void* weights_ptr = &weights;
    
    // Initialize if first time
    if (m_map.find(weights_ptr) == m_map.end()) {
        m_map[weights_ptr] = Tensor(weights.get_dim4(), weights.get_dim3(), weights.get_dim2(), weights.get_dim1());
        v_map[weights_ptr] = Tensor(weights.get_dim4(), weights.get_dim3(), weights.get_dim2(), weights.get_dim1());
        m_map[weights_ptr].zero();
        v_map[weights_ptr].zero();
        step_count_map[weights_ptr] = 0;
    }
    
    step_count_map[weights_ptr]++;
    int t = step_count_map[weights_ptr];
    
    Tensor& m = m_map[weights_ptr];  
    Tensor& v = v_map[weights_ptr];  

    m *= params.beta1;
    m += gradients * (1.0 - params.beta1);

    Tensor grad_squared = gradients * gradients;
    v *= params.beta2;
    v += grad_squared * (1.0 - params.beta2);

    double bias_correction1 = 1.0 - pow(params.beta1, t);
    double bias_correction2 = 1.0 - pow(params.beta2, t);
    
    Tensor m_hat = m * (1.0 / bias_correction1);
    Tensor v_hat = v * (1.0 / bias_correction2);

    Tensor v_sqrt = v_hat.sqrt();
    v_sqrt += params.epsilon;
    Tensor update = m_hat / v_sqrt * params.learning_rate;
    weights -= update;
}

Optimizer create_sgd(double lr) {
    return Optimizer(OptimizerType::SGD, OptimizerParams::sgd(lr));
}

Optimizer create_adam(double lr) {
    return Optimizer(OptimizerType::ADAM, OptimizerParams::adam(lr));
}

Optimizer create_sgd_momentum(double lr, double momentum) {
    return Optimizer(OptimizerType::SGD_MOMENTUM, OptimizerParams::sgd_momentum(lr, momentum));
}

Optimizer create_rmsprop(double lr, double decay) {
    return Optimizer(OptimizerType::RMSPROP, OptimizerParams::rmsprop(lr, decay));
}