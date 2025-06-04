#include "..\headers\Loss.hpp"

double Loss::forward(const Tensor& predictions, const Tensor& targets, LossType type) {
    validate_shapes(predictions, targets);
    
    switch (type) {
        case LossType::MEAN_SQUARED_ERROR:
            return mean_squared_error(predictions, targets);
        case LossType::CROSS_ENTROPY:
            return cross_entropy(predictions, targets);
        case LossType::BINARY_CROSS_ENTROPY:
            return binary_cross_entropy(predictions, targets);
        case LossType::MEAN_ABSOLUTE_ERROR:
            return mean_absolute_error(predictions, targets);
        default:
            throw invalid_argument("Unknown loss type");
    }
}

Tensor Loss::backward(const Tensor& predictions, const Tensor& targets, LossType type) {
    validate_shapes(predictions, targets);
    
    switch (type) {
        case LossType::MEAN_SQUARED_ERROR:
            return mean_squared_error_backward(predictions, targets);
        case LossType::CROSS_ENTROPY:
            return cross_entropy_backward(predictions, targets);
        case LossType::BINARY_CROSS_ENTROPY:
            return binary_cross_entropy_backward(predictions, targets);
        case LossType::MEAN_ABSOLUTE_ERROR:
            return mean_absolute_error_backward(predictions, targets);
        default:
            throw invalid_argument("Unknown loss type");
    }
}

double Loss::mean_squared_error(const Tensor& predictions, const Tensor& targets) {
    double total_loss = 0.0;
    int total_elements = predictions.get_size();
    
    const double* pred_data = predictions.get_data();
    const double* target_data = targets.get_data();
    
    for (int i = 0; i < total_elements; i++) {
        double diff = pred_data[i] - target_data[i];
        total_loss += diff * diff;
    }
    
    return total_loss / total_elements; 
}

Tensor Loss::mean_squared_error_backward(const Tensor& predictions, const Tensor& targets) {
    Tensor gradient(predictions.get_dim4(), predictions.get_dim3(), predictions.get_dim2(), predictions.get_dim1());
    
    int total_elements = predictions.get_size();
    const double* pred_data = predictions.get_data();
    const double* target_data = targets.get_data();
    double* grad_data = gradient.get_data();
    
    for (int i = 0; i < total_elements; i++) {
        grad_data[i] = 2.0 * (pred_data[i] - target_data[i]) / total_elements;
    }
    
    return gradient;
}

double Loss::cross_entropy(const Tensor& predictions, const Tensor& targets) {
    double total_loss = 0.0;
    int batch_size = predictions.get_dim4();
    int channels = predictions.get_dim3();
    int height = predictions.get_dim2();
    int num_classes = predictions.get_dim1();
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int i = 0; i < num_classes; i++) {
                    double target = targets(b, c, h, i);
                    double prediction = predictions(b, c, h, i);
                    
                    if (target > 0.0) {  
                        total_loss -= target * safe_log(prediction);
                    }
                }
            }
        }
    }
    
    int total_samples = batch_size * channels * height;
    return total_loss / total_samples;
}

Tensor Loss::cross_entropy_backward(const Tensor& predictions, const Tensor& targets) {
    Tensor gradient(predictions.get_dim4(), predictions.get_dim3(), predictions.get_dim2(), predictions.get_dim1());
    
    int batch_size = predictions.get_dim4();
    int channels = predictions.get_dim3();
    int height = predictions.get_dim2();
    int num_classes = predictions.get_dim1();
    int total_samples = batch_size * channels * height;
    
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int i = 0; i < num_classes; i++) {
                    gradient(b, c, h, i) = (predictions(b, c, h, i) - targets(b, c, h, i)) / total_samples;
                }
            }
        }
    }
    
    return gradient;
}

double Loss::binary_cross_entropy(const Tensor& predictions, const Tensor& targets) {
    double total_loss = 0.0;
    int total_elements = predictions.get_size();
    
    const double* pred_data = predictions.get_data();
    const double* target_data = targets.get_data();
    
    for (int i = 0; i < total_elements; i++) {
        double pred = pred_data[i];
        double target = target_data[i];
        
        total_loss -= target * safe_log(pred) + (1.0 - target) * safe_log(1.0 - pred);
    }
    
    return total_loss / total_elements;
}

Tensor Loss::binary_cross_entropy_backward(const Tensor& predictions, const Tensor& targets) {
    Tensor gradient(predictions.get_dim4(), predictions.get_dim3(), predictions.get_dim2(), predictions.get_dim1());
    
    int total_elements = predictions.get_size();
    const double* pred_data = predictions.get_data();
    const double* target_data = targets.get_data();
    double* grad_data = gradient.get_data();
    
    for (int i = 0; i < total_elements; i++) {
        grad_data[i] = (pred_data[i] - target_data[i]) / total_elements;
    }
    
    return gradient;
}

double Loss::mean_absolute_error(const Tensor& predictions, const Tensor& targets) {
    double total_loss = 0.0;
    int total_elements = predictions.get_size();
    
    const double* pred_data = predictions.get_data();
    const double* target_data = targets.get_data();
    
    for (int i = 0; i < total_elements; i++) {
        total_loss += abs(pred_data[i] - target_data[i]);
    }
    
    return total_loss / total_elements;
}

Tensor Loss::mean_absolute_error_backward(const Tensor& predictions, const Tensor& targets) {
    Tensor gradient(predictions.get_dim4(), predictions.get_dim3(), predictions.get_dim2(), predictions.get_dim1());
    
    int total_elements = predictions.get_size();
    const double* pred_data = predictions.get_data();
    const double* target_data = targets.get_data();
    double* grad_data = gradient.get_data();
    
    for (int i = 0; i < total_elements; i++) {
        double diff = pred_data[i] - target_data[i];
        if (diff > 0) {
            grad_data[i] = 1.0 / total_elements;
        } else if (diff < 0) {
            grad_data[i] = -1.0 / total_elements;
        } else {
            grad_data[i] = 0.0; 
        }
    }
    
    return gradient;
}

string Loss::get_loss_name(LossType type) {
    switch (type) {
        case LossType::MEAN_SQUARED_ERROR: return "Mean Squared Error";
        case LossType::CROSS_ENTROPY: return "Cross Entropy";
        case LossType::BINARY_CROSS_ENTROPY: return "Binary Cross Entropy";
        case LossType::MEAN_ABSOLUTE_ERROR: return "Mean Absolute Error";
        default: return "Unknown Loss";
    }
}

double Loss::safe_log(double x) {
    const double epsilon = 1e-15;
    return log(max(x, epsilon));
}

void Loss::validate_shapes(const Tensor& predictions, const Tensor& targets) {
    if (predictions.get_dim4() != targets.get_dim4() ||
        predictions.get_dim3() != targets.get_dim3() ||
        predictions.get_dim2() != targets.get_dim2() ||
        predictions.get_dim1() != targets.get_dim1()) {
        
        throw invalid_argument("Predictions and targets must have the same shape");
    }
}

int Loss::get_batch_size(const Tensor& tensor) {
    return tensor.get_dim4();
}