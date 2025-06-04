#include "..\headers\Batch_Normalization.hpp"

BatchNormalization::BatchNormalization(int features, BatchNormType type, double eps, double mom, const string& name) : Layer(name, true), norm_type(type), num_features(features), epsilon(eps), momentum(mom), training_mode(true) {
    
    if (features <= 0) {
        throw invalid_argument("Number of features must be positive");
    }
    if (eps <= 0) {
        throw invalid_argument("Epsilon must be positive");
    }
    if (mom < 0 || mom > 1) {
        throw invalid_argument("Momentum must be between 0 and 1");
    }
    
    gamma = Tensor(1, 1, 1, num_features);
    beta = Tensor(1, 1, 1, num_features);
    gamma_gradients = Tensor(1, 1, 1, num_features);
    beta_gradients = Tensor(1, 1, 1, num_features);

    running_mean = Tensor(1, 1, 1, num_features);
    running_var = Tensor(1, 1, 1, num_features);

    batch_mean = Tensor(1, 1, 1, num_features);
    batch_var = Tensor(1, 1, 1, num_features);
    
    initialize_weights();
}

BatchNormalization::~BatchNormalization() {}

void BatchNormalization::initialize_weights() {
    gamma.fill(1.0);
    beta.zero();
    
    running_mean.zero();
    running_var.fill(1.0);
}

void BatchNormalization::calculate_batch_stats(const Tensor& input) {
    int batch_size = input.get_dim4();
    
    batch_mean.zero();
    batch_var.zero();
    
    if (norm_type == BatchNormType::DENSE) {
        
        for (int f = 0; f < num_features; f++) {

            double sum = 0.0;
            for (int b = 0; b < batch_size; b++) {
                sum += input(b, 0, 0, f);
            }
            double mean = sum / batch_size;
            batch_mean(0, 0, 0, f) = mean;
            
            double var_sum = 0.0;
            for (int b = 0; b < batch_size; b++) {
                double diff = input(b, 0, 0, f) - mean;
                var_sum += diff * diff;
            }
            double variance = var_sum / batch_size;
            batch_var(0, 0, 0, f) = variance;
        }
        
    } else { // CONV2D
        
        int height = input.get_dim2();
        int width = input.get_dim1();
        int spatial_size = height * width;
        int total_elements = batch_size * spatial_size;

        const double* input_data = input.get_data();
        
        for (int c = 0; c < num_features; c++) {
            
            double sum = 0.0;

            for (int b = 0; b < batch_size; b++) {
                int channel_base = ((b * num_features + c) * height * width);
                for (int spatial = 0; spatial < spatial_size; spatial++) {
                    sum += input_data[channel_base + spatial];
                }
            }
            double mean = sum / total_elements;
            batch_mean(0, 0, 0, c) = mean;

            double var_sum = 0.0;
            for (int b = 0; b < batch_size; b++) {
                int channel_base = ((b * num_features + c) * height * width);
                for (int spatial = 0; spatial < spatial_size; spatial++) {
                    double diff = input_data[channel_base + spatial] - mean;
                    var_sum += diff * diff;
                }
            }
            double variance = var_sum / total_elements;
            batch_var(0, 0, 0, c) = variance;
        }
    }
}

void BatchNormalization::update_running_stats() {
    for (int f = 0; f < num_features; f++) {
        double current_running_mean = running_mean(0, 0, 0, f);
        double current_running_var = running_var(0, 0, 0, f);
        double current_batch_mean = batch_mean(0, 0, 0, f);
        double current_batch_var = batch_var(0, 0, 0, f);
        
        running_mean(0, 0, 0, f) = (1.0 - momentum) * current_running_mean + momentum * current_batch_mean;
        running_var(0, 0, 0, f) = (1.0 - momentum) * current_running_var + momentum * current_batch_var;
    }
}

Tensor BatchNormalization::normalize_input(const Tensor& input) {
    Tensor output(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    if (norm_type == BatchNormType::DENSE) {
        int batch_size = input.get_dim4();
        
        for (int b = 0; b < batch_size; b++) {
            for (int f = 0; f < num_features; f++) {
                double mean = training_mode ? batch_mean(0, 0, 0, f) : running_mean(0, 0, 0, f);
                double var = training_mode ? batch_var(0, 0, 0, f) : running_var(0, 0, 0, f);
                
                double normalized = (input(b, 0, 0, f) - mean) / sqrt(var + epsilon);

                output(b, 0, 0, f) = gamma(0, 0, 0, f) * normalized + beta(0, 0, 0, f);
            }
        }
        
    } else { // CONV2D
        int batch_size = input.get_dim4();
        int height = input.get_dim2();
        int width = input.get_dim1();
        
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < num_features; c++) {
                double mean = training_mode ? batch_mean(0, 0, 0, c) : running_mean(0, 0, 0, c);
                double var = training_mode ? batch_var(0, 0, 0, c) : running_var(0, 0, 0, c);
                
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        double normalized = (input(b, c, h, w) - mean) / sqrt(var + epsilon);
                        
                        output(b, c, h, w) = gamma(0, 0, 0, c) * normalized + beta(0, 0, 0, c);
                    }
                }
            }
        }
    }
    
    return output;
}

Tensor BatchNormalization::forward(const Tensor& input) {
    this->input = input;
    
    if (norm_type == BatchNormType::DENSE) {
        if (input.get_dim1() != num_features) {
            throw invalid_argument("Input feature dimension mismatch for Dense BatchNorm");
        }
    } else { // CONV2D
        if (input.get_dim3() != num_features) {
            throw invalid_argument("Input channel dimension mismatch for Conv2D BatchNorm");
        }
    }
    
    if (training_mode) {
        calculate_batch_stats(input);
        
        normalized_input = Tensor(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
        
        // normalized values before scale and shift
        if (norm_type == BatchNormType::DENSE) {
            int batch_size = input.get_dim4();
            for (int b = 0; b < batch_size; b++) {
                for (int f = 0; f < num_features; f++) {
                    double mean = batch_mean(0, 0, 0, f);
                    double var = batch_var(0, 0, 0, f);
                    normalized_input(b, 0, 0, f) = (input(b, 0, 0, f) - mean) / sqrt(var + epsilon);
                }
            }
        } else { // CONV2D
            int batch_size = input.get_dim4();
            int height = input.get_dim2();
            int width = input.get_dim1();
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < num_features; c++) {
                    double mean = batch_mean(0, 0, 0, c);
                    double var = batch_var(0, 0, 0, c);
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            normalized_input(b, c, h, w) = (input(b, c, h, w) - mean) / sqrt(var + epsilon);
                        }
                    }
                }
            }
        }
        
        update_running_stats();
    }
    
    this->output = normalize_input(input);
    
    return this->output;
}

Tensor BatchNormalization::backward(const Tensor& gradient) {
    if (!training_mode) {
        // scaling gradients by gamma 
        Tensor input_gradient(gradient.get_dim4(), gradient.get_dim3(), gradient.get_dim2(), gradient.get_dim1());
        
        const double* grad_data = gradient.get_data();
        const double* gamma_data = gamma.get_data();
        const double* running_var_data = running_var.get_data();
        double* input_grad_data = input_gradient.get_data();
        
        if (norm_type == BatchNormType::DENSE) {
            int batch_size = gradient.get_dim4();
            for (int b = 0; b < batch_size; b++) {
                for (int f = 0; f < num_features; f++) {
                    double var = running_var_data[f];
                    double scale = gamma_data[f] / sqrt(var + epsilon);
                    
                    int idx = b * num_features + f;
                    input_grad_data[idx] = grad_data[idx] * scale;
                }
            }
        } else { // CONV2D
            int batch_size = gradient.get_dim4();
            int height = gradient.get_dim2();
            int width = gradient.get_dim1();
            int spatial_size = height * width;
            
            for (int b = 0; b < batch_size; b++) {
                for (int c = 0; c < num_features; c++) {
                    double var = running_var_data[c];
                    double scale = gamma_data[c] / sqrt(var + epsilon);
                    
                    int channel_base = ((b * num_features + c) * height * width);
                    for (int spatial = 0; spatial < spatial_size; spatial++) {
                        input_grad_data[channel_base + spatial] = grad_data[channel_base + spatial] * scale;
                    }
                }
            }
        }
        
        return input_gradient;
    }

    // Training mode
    gamma_gradients.zero();
    beta_gradients.zero();
    
    int batch_size = input.get_dim4();
    Tensor input_gradient(input.get_dim4(), input.get_dim3(), input.get_dim2(), input.get_dim1());
    
    const double* grad_data = gradient.get_data();
    const double* normalized_data = normalized_input.get_data();
    const double* gamma_data = gamma.get_data();
    const double* batch_var_data = batch_var.get_data();
    double* gamma_grad_data = gamma_gradients.get_data();
    double* beta_grad_data = beta_gradients.get_data();
    double* input_grad_data = input_gradient.get_data();
    
    if (norm_type == BatchNormType::DENSE) {
        // Compute gamma and beta gradients
        for (int f = 0; f < num_features; f++) {
            double gamma_grad = 0.0;
            double beta_grad = 0.0;
            
            for (int b = 0; b < batch_size; b++) {
                int idx = b * num_features + f;
                gamma_grad += grad_data[idx] * normalized_data[idx];
                beta_grad += grad_data[idx];
            }
            
            gamma_grad_data[f] = gamma_grad;
            beta_grad_data[f] = beta_grad;
        }
        
        // Compute input gradients
        for (int f = 0; f < num_features; f++) {
            double var = batch_var_data[f];
            double std_inv = 1.0 / sqrt(var + epsilon);
            double gamma_val = gamma_data[f];
            
            double grad_norm = 0.0;
            double grad_mean = 0.0;
            
            // Accumulate gradients
            for (int b = 0; b < batch_size; b++) {
                int idx = b * num_features + f;
                grad_norm += grad_data[idx] * normalized_data[idx];
                grad_mean += grad_data[idx];
            }
            
            // Apply gradients
            for (int b = 0; b < batch_size; b++) {
                int idx = b * num_features + f;
                input_grad_data[idx] = gamma_val * std_inv / batch_size * 
                    (batch_size * grad_data[idx] - grad_mean - normalized_data[idx] * grad_norm);
            }
        }
        
    } else { // CONV2D 
        int height = input.get_dim2();
        int width = input.get_dim1();
        int spatial_size = height * width;
        int total_elements = batch_size * spatial_size;
        
        // Compute gamma and beta gradients
        for (int c = 0; c < num_features; c++) {
            double gamma_grad = 0.0;
            double beta_grad = 0.0;
            
            for (int b = 0; b < batch_size; b++) {
                int channel_base = ((b * num_features + c) * height * width);
                for (int spatial = 0; spatial < spatial_size; spatial++) {
                    int idx = channel_base + spatial;
                    gamma_grad += grad_data[idx] * normalized_data[idx];
                    beta_grad += grad_data[idx];
                }
            }
            
            gamma_grad_data[c] = gamma_grad;
            beta_grad_data[c] = beta_grad;
        }
        
        // Compute input gradients
        for (int c = 0; c < num_features; c++) {
            double var = batch_var_data[c];
            double std_inv = 1.0 / sqrt(var + epsilon);
            double gamma_val = gamma_data[c];
            
            double grad_norm = 0.0;
            double grad_mean = 0.0;
            
            // Accumulate gradients across all spatial locations
            for (int b = 0; b < batch_size; b++) {
                int channel_base = ((b * num_features + c) * height * width);
                for (int spatial = 0; spatial < spatial_size; spatial++) {
                    int idx = channel_base + spatial;
                    grad_norm += grad_data[idx] * normalized_data[idx];
                    grad_mean += grad_data[idx];
                }
            }
            
            // Apply gradients to all spatial locations
            for (int b = 0; b < batch_size; b++) {
                int channel_base = ((b * num_features + c) * height * width);
                for (int spatial = 0; spatial < spatial_size; spatial++) {
                    int idx = channel_base + spatial;
                    input_grad_data[idx] = gamma_val * std_inv / total_elements * 
                        (total_elements * grad_data[idx] - grad_mean - normalized_data[idx] * grad_norm);
                }
            }
        }
    }
    
    return input_gradient;
}

void BatchNormalization::update_weights(double learning_rate) {
    gamma -= gamma_gradients * learning_rate;
    beta -= beta_gradients * learning_rate;
}

void BatchNormalization::set_training(bool training) {
    training_mode = training;
}

vector<int> BatchNormalization::output_shape(const vector<int>& input_shape) const {
    return input_shape;
}

int BatchNormalization::param_count() const {
    return 2 * num_features; // gamma + beta
}

const Tensor& BatchNormalization::get_gamma() const {return gamma;}
const Tensor& BatchNormalization::get_beta() const {return beta;}
const Tensor& BatchNormalization::get_running_mean() const {return running_mean;}
const Tensor& BatchNormalization::get_running_var() const {return running_var;}
bool BatchNormalization::is_training() const {return training_mode;}

void BatchNormalization::set_gamma(const Tensor& new_gamma) {
    if (new_gamma.get_dim1() != num_features) {
        throw invalid_argument("Gamma dimensions don't match number of features");
    }
    gamma = new_gamma;
}

void BatchNormalization::set_beta(const Tensor& new_beta) {
    if (new_beta.get_dim1() != num_features) {
        throw invalid_argument("Beta dimensions don't match number of features");
    }
    beta = new_beta;
}