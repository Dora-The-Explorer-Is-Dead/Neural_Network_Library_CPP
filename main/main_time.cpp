#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <iomanip>

// Include your headers
#include "../headers/Tensor.hpp"
#include "../headers/Dense_Layer.hpp"
#include "../headers/Conv2D_Layer.hpp"
#include "../headers/Pooling.hpp"
#include "../headers/Flatten.hpp"
#include "../headers/Activation.hpp"
#include "../headers/Loss.hpp"
#include "../headers/Optimizer.hpp"
#include "../headers/Batch_Normalization.hpp"
#include "../headers/Dropout_Layer.hpp"

using namespace std;
using namespace std::chrono;

const int TRAIN_SAMPLES = 3200;   
const int TEST_SAMPLES = 1600;    
const int BATCH_SIZE = 64;
const int EPOCHS = 12;
const double LEARNING_RATE = 0.001;
const int IMAGE_SIZE = 28;
const int NUM_CLASSES = 10;

pair<Tensor, Tensor> load_mnist_tensors(const string& filename, int max_samples) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    // images (N, 1, 28, 28), labels (N, 1, 1, 10) one-hot
    Tensor images(max_samples, 1, IMAGE_SIZE, IMAGE_SIZE);
    Tensor labels(max_samples, 1, 1, NUM_CLASSES);
    labels.zero();
    
    string line;
    getline(file, line); 
    
    int sample_idx = 0;
    while (getline(file, line) && sample_idx < max_samples) {
        stringstream ss(line);
        string value;
        
        // label 
        getline(ss, value, ',');
        int label = stoi(value);
        labels(sample_idx, 0, 0, label) = 1.0; // One-hot encoding
        
        // pixels (normalize to [0,1])
        int pixel_idx = 0;
        while (getline(ss, value, ',') && pixel_idx < 784) {
            double pixel = stod(value) / 255.0;
            int row = pixel_idx / IMAGE_SIZE;
            int col = pixel_idx % IMAGE_SIZE;
            images(sample_idx, 0, row, col) = pixel;
            pixel_idx++;
        }
        
        sample_idx++;
        if (sample_idx % 1000 == 0) {
            cout << "Loaded " << sample_idx << " samples..." << endl;
        }
    }
    
    file.close();
    cout << "Successfully loaded " << sample_idx << " samples from " << filename << endl;
    return {images, labels};
}

// Extract batch from larger tensor
Tensor extract_batch(const Tensor& data, int start_idx, int batch_size) {
    int actual_batch_size = min(batch_size, data.get_dim4() - start_idx);
    Tensor batch(actual_batch_size, data.get_dim3(), data.get_dim2(), data.get_dim1());
    
    for (int b = 0; b < actual_batch_size; b++) {
        for (int c = 0; c < data.get_dim3(); c++) {
            for (int h = 0; h < data.get_dim2(); h++) {
                for (int w = 0; w < data.get_dim1(); w++) {
                    batch(b, c, h, w) = data(start_idx + b, c, h, w);
                }
            }
        }
    }
    
    return batch;
}

// Calculate accuracy from prediction and label tensors
double calculate_accuracy(const Tensor& predictions, const Tensor& true_labels) {
    int batch_size = predictions.get_dim4();
    int correct = 0;
    
    for (int b = 0; b < batch_size; b++) {
        // Find predicted class (max probability)
        int predicted_class = 0;
        double max_prob = predictions(b, 0, 0, 0);
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (predictions(b, 0, 0, c) > max_prob) {
                max_prob = predictions(b, 0, 0, c);
                predicted_class = c;
            }
        }
        
        // Find true class (one-hot encoded)
        int true_class = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (true_labels(b, 0, 0, c) > 0.5) {
                true_class = c;
                break;
            }
        }
        
        if (predicted_class == true_class) {
            correct++;
        }
    }
    
    return (double)correct / batch_size;
}

// CNN 
class MNISTNet {
private:
    Conv2DLayer conv1;
    BatchNormalization bn1;
    PoolingLayer pool1;
    
    Conv2DLayer conv2;
    BatchNormalization bn2;
    PoolingLayer pool2;
    
    FlattenLayer flatten;
    DenseLayer fc1;
    BatchNormalization bn3;
    DropoutLayer dropout;
    DenseLayer fc2;
    
    Optimizer optimizer;
    
    // Store intermediate tensors for proper backprop
    Tensor conv1_out, bn1_out, relu1_out, pool1_out;
    Tensor conv2_out, bn2_out, relu2_out, pool2_out;
    Tensor flat_out, fc1_out, bn3_out, relu3_out, drop_out;
    
public:
    MNISTNet() : 
        conv1(1, 32, 3, 1, ActivationType::RELU, "conv1"),
        bn1(32, BatchNormType::CONV2D, 1e-5, 0.9, "bn1"),
        pool1(PoolingType::MAX, 2, 2, "pool1"),
        
        conv2(32, 64, 3, 1, ActivationType::RELU, "conv2"),
        bn2(64, BatchNormType::CONV2D, 1e-5, 0.9, "bn2"),
        pool2(PoolingType::MAX, 2, 2, "pool2"),
        
        flatten("flatten"),
        fc1(64 * 7 * 7, 128, ActivationType::RELU, "fc1"),
        bn3(128, BatchNormType::DENSE, 1e-5, 0.9, "bn3"),
        dropout(DropoutRate::MODERATE, "dropout"),
        fc2(128, NUM_CLASSES, ActivationType::RELU, "fc2"),
        
        optimizer(create_adam(LEARNING_RATE))
    {}
    
    Tensor forward(const Tensor& input, bool training = true) {
        // Set training modes
        bn1.set_training(training);
        bn2.set_training(training);
        bn3.set_training(training);
        dropout.set_training(training);
        
        // Forward pass with stored intermediates
        conv1_out = conv1.forward(input);
        bn1_out = bn1.forward(conv1_out);
        relu1_out = Activation::forward(bn1_out, ActivationType::RELU);
        pool1_out = pool1.forward(relu1_out);
        
        conv2_out = conv2.forward(pool1_out);
        bn2_out = bn2.forward(conv2_out);
        relu2_out = Activation::forward(bn2_out, ActivationType::RELU);
        pool2_out = pool2.forward(relu2_out);
        
        flat_out = flatten.forward(pool2_out);
        fc1_out = fc1.forward(flat_out);
        bn3_out = bn3.forward(fc1_out);
        relu3_out = Activation::forward(bn3_out, ActivationType::RELU);
        drop_out = dropout.forward(relu3_out);
        Tensor fc2_out = fc2.forward(drop_out);
        
        // Softmax for final output
        return Activation::forward(fc2_out, ActivationType::SOFTMAX);
    }
    
    void backward(const Tensor& gradient) {
        // Backward through softmax (gradient already computed by loss function)
        Tensor grad = gradient;
        
        // Backward pass using stored forward tensors
        grad = fc2.backward(grad);
        grad = dropout.backward(grad);
        grad = Activation::backward(bn3_out, grad, ActivationType::RELU);
        grad = bn3.backward(grad);
        grad = fc1.backward(grad);
        grad = flatten.backward(grad);
        
        grad = pool2.backward(grad);
        grad = Activation::backward(bn2_out, grad, ActivationType::RELU);
        grad = bn2.backward(grad);
        grad = conv2.backward(grad);
        
        grad = pool1.backward(grad);
        grad = Activation::backward(bn1_out, grad, ActivationType::RELU);
        grad = bn1.backward(grad);
        conv1.backward(grad);
    }
    
    void update_weights() {
        double lr = optimizer.get_learning_rate();
        conv1.update_weights(lr);
        conv2.update_weights(lr);
        fc1.update_weights(lr);
        fc2.update_weights(lr);
        bn1.update_weights(lr);
        bn2.update_weights(lr);
        bn3.update_weights(lr);
    }
};

int main() {
    cout << "=== MNIST CNN Training with Pure Tensors ===" << endl;
    cout << "Config: " << TRAIN_SAMPLES << " train, " << TEST_SAMPLES << " test, " 
         << BATCH_SIZE << " batch, " << EPOCHS << " epochs" << endl << endl;
    
    try {
        cout << "Loading training data..." << endl;
        auto [train_images, train_labels] = load_mnist_tensors("C:/UMAMA MUHAMMAD/OOP/OOP_PROJECT_REDO_LESSSGOOO/mnist_train.csv", TRAIN_SAMPLES);
        
        cout << "Loading test data..." << endl;
        auto [test_images, test_labels] = load_mnist_tensors("C:/UMAMA MUHAMMAD/OOP/OOP_PROJECT_REDO_LESSSGOOO/mnist_test.csv", TEST_SAMPLES);

        cout << "Initializing CNN..." << endl;
        MNISTNet network;
        cout << "Network ready!" << endl << endl;

        cout << "Starting training..." << endl;
        cout << string(70, '=') << endl;
        
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            auto epoch_start = high_resolution_clock::now();
            
            double total_loss = 0.0;
            double total_accuracy = 0.0;
            int num_batches = 0;

            for (int i = 0; i < TRAIN_SAMPLES; i += BATCH_SIZE) {
                // Extract batch tensors
                Tensor batch_images = extract_batch(train_images, i, BATCH_SIZE);
                Tensor batch_labels = extract_batch(train_labels, i, BATCH_SIZE);
                
                auto total_batch_start = high_resolution_clock::now();

                // Time forward pass
                auto forward_start = high_resolution_clock::now();
                Tensor predictions = network.forward(batch_images, true);
                auto forward_end = high_resolution_clock::now();

                // Time loss calculation
                auto loss_start = high_resolution_clock::now();
                double batch_loss = Loss::forward(predictions, batch_labels, LossType::CROSS_ENTROPY);
                double batch_accuracy = calculate_accuracy(predictions, batch_labels);
                auto loss_end = high_resolution_clock::now();

                total_loss += batch_loss;
                total_accuracy += batch_accuracy;
                num_batches++;

                // Time backward pass
                auto backward_start = high_resolution_clock::now();
                Tensor loss_gradient = Loss::backward(predictions, batch_labels, LossType::CROSS_ENTROPY);
                network.backward(loss_gradient);
                auto backward_end = high_resolution_clock::now();

                // Time weight update
                auto update_start = high_resolution_clock::now();
                network.update_weights();
                auto update_end = high_resolution_clock::now();

                auto total_batch_end = high_resolution_clock::now();

                // Print timing for first batch only
                if (num_batches == 1) {
                    cout << "\n=== TIMING BREAKDOWN (First Batch) ===" << endl;
                    cout << "Forward:  " << duration_cast<milliseconds>(forward_end - forward_start).count() << "ms" << endl;
                    cout << "Loss:     " << duration_cast<milliseconds>(loss_end - loss_start).count() << "ms" << endl;
                    cout << "Backward: " << duration_cast<milliseconds>(backward_end - backward_start).count() << "ms" << endl;
                    cout << "Update:   " << duration_cast<milliseconds>(update_end - update_start).count() << "ms" << endl;
                    cout << "TOTAL:    " << duration_cast<milliseconds>(total_batch_end - total_batch_start).count() << "ms" << endl;
                    cout << "=========================================\n" << endl;
                }
                
                // Progress dots
                if (num_batches % 10 == 0) cout << "." << flush;
            }
            
            auto epoch_end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(epoch_end - epoch_start);
            
            // Print epoch results
            double avg_loss = total_loss / num_batches;
            double avg_accuracy = total_accuracy / num_batches;
            
            cout << endl << "Epoch " << setw(2) << epoch + 1 << "/" << EPOCHS 
                 << " | Loss: " << fixed << setprecision(4) << avg_loss
                 << " | Train Acc: " << setprecision(1) << avg_accuracy * 100 << "%"
                 << " | Time: " << duration.count() << "ms" << endl;
        }
        
        cout << string(70, '=') << endl;
        
        // Test evaluation
        cout << "Evaluating on test set..." << endl;
        double test_accuracy = 0.0;
        int test_batches = 0;
        
        for (int i = 0; i < TEST_SAMPLES; i += BATCH_SIZE) {
            Tensor batch_images = extract_batch(test_images, i, BATCH_SIZE);
            Tensor batch_labels = extract_batch(test_labels, i, BATCH_SIZE);
            
            // Forward pass in evaluation mode
            Tensor predictions = network.forward(batch_images, false);
            double batch_accuracy = calculate_accuracy(predictions, batch_labels);
            
            test_accuracy += batch_accuracy;
            test_batches++;
        }
        
        test_accuracy /= test_batches;
        
        cout << "\n🎯 Final Test Accuracy: " << fixed << setprecision(1) << test_accuracy * 100 << "%" << endl;
        
        if (test_accuracy > 0.95) {
            cout << "🎉 Excellent performance! >95% accuracy achieved!" << endl;
        } else if (test_accuracy > 0.90) {
            cout << "✅ Good performance! >90% accuracy achieved!" << endl;
        } else {
            cout << "📈 Training complete." << endl;
        }
        
    } catch (const exception& e) {
        cerr << "❌ Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}