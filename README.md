# Neural Network Library C++

A lightweight, efficient neural network library implemented in C++. This library provides a clean, object-oriented approach to building and training neural networks from scratch.

## → Detailed File Structure

```
Neural_Network_Library_CPP/
├── src/
│   ├── Activation.cpp             # Activation functions (ReLU, Sigmoid, Tanh, Softmax, Leaky ReLU)
│   ├── Batch_Normalization.cpp    # Batch normalization for training stability
│   ├── Conv2D_Layer.cpp           # 2D Convolutional layers with im2col optimization
│   ├── Dense_Layer.cpp            # Fully connected/dense layers
│   ├── Dropout_Layer.cpp          # Dropout regularization
│   ├── Flatten.cpp                # Flatten layer for CNN-to-Dense transitions
│   ├── Initializer_Weights_Biases.cpp # Weight initialization (Xavier, He, etc.)
│   ├── Layer.cpp                  # Base layer abstract class
│   ├── Loss.cpp                   # Loss functions (MSE, Cross-entropy, Binary CE, MAE)
│   ├── Optimizer.cpp              # Optimizers (SGD, Adam, RMSprop, SGD+Momentum)
│   ├── Pooling.cpp                # Pooling layers (Max, Min, Average pooling)
│   └── Tensor.cpp                 # Custom 4D tensor implementation with broadcasting
├── headers/                       # Header files (.hpp files)
├── main/                          # Example programs and demos
├── .vscode/                       # VS Code configuration
├── .gitignore                     # Git ignore file (excludes .csv, .exe files)
└── README.md                      # This documentation
```

## → Installation

### Prerequisites

- **C++ Compiler** (C++11 or later)
  - GCC 4.8+ on Linux
  - MSVC 2015+ on Windows
  - Clang 3.4+ on macOS

### Building the Project

#### Using g++
```bash
# Clone the repository
git clone https://github.com/Dora-The-Explorer-Is-Dead/Neural_Network_Library_CPP.git
cd Neural_Network_Library_CPP

# Compile example programs
g++ -O2 -std=c++11 -I./include main/main.cpp src/*.cpp -o neural_network_demo
```

### Dataset Setup

This project uses the MNIST dataset for training and testing. Download the CSV files:

1. **Download MNIST CSV files:**
   - https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download

2. **Place them in the project root directory:**
   ```
   Neural_Network_Library_CPP/
   ├── mnist_train.csv
   ├── mnist_test.csv
   └── ...
   ```

## → Performance & Capabilities

### Supported Architectures
- **Dense Networks**: Fully connected layers with any activation
- **Convolutional Networks**: 2D convolutions with pooling and batch norm
- **Hybrid Architectures**: Mix CNN and dense layers seamlessly
- **Regularization**: Dropout and batch normalization for better generalization

### Optimization Algorithms
- **SGD**: Basic gradient descent with configurable learning rate
- **SGD + Momentum**: Accelerated convergence with momentum parameter
- **Adam**: Adaptive learning rates with first and second moment estimates
- **RMSprop**: Adaptive learning rate scaling

### Tensor Operations
- **Broadcasting**: Automatic dimension handling for operations
- **Matrix Multiplication**: Optimized BLAS-style implementations
- **Memory Efficiency**: In-place operations where possible
- **4D Support**: Native support for batch processing

## → Core Components Explained

### Tensor.cpp - Custom 4D Tensor Implementation
- **4D Tensor Structure**: (batch, channels, height, width) for both images and dense data
- **Matrix Operations**: Optimized matrix multiplication with broadcasting
- **Element-wise Operations**: Addition, subtraction, multiplication, division
- **Advanced Functions**: Transpose, reshape, sum operations
- **Memory Management**: Efficient data handling with move semantics

### Layer System Architecture

#### Dense_Layer.cpp - Fully Connected Layers
- **Forward Pass**: Matrix multiplication with bias addition
- **Backward Pass**: Gradient computation for weights and biases
- **Weight Initialization**: Automatic selection based on activation type
- **Flexible Input**: Handles any tensor shape (flattened internally)

#### Conv2D_Layer.cpp - Convolutional Layers
- **Im2col Optimization**: Converts convolution to matrix multiplication
- **Automatic Padding**: Maintains spatial dimensions
- **Configurable Parameters**: Kernel size, stride, channels
- **Memory Efficient**: Optimized data access patterns

#### Pooling.cpp - Pooling Operations
- **Multiple Types**: Max, Min, and Average pooling
- **Automatic Padding**: Handles edge cases gracefully
- **Index Tracking**: Remembers max/min positions for backpropagation
- **Flexible Stride**: Configurable stride patterns

#### Batch_Normalization.cpp - Training Stabilization
- **Batch Statistics**: Computes mean and variance per batch
- **Running Statistics**: Maintains moving averages for inference
- **Dual Mode**: Separate behavior for training vs inference
- **Channel-wise**: Normalizes across spatial dimensions

#### Dropout_Layer.cpp - Regularization
- **Inverted Dropout**: Scales activations during training
- **Preset Rates**: Predefined dropout levels (Light, Moderate, Heavy)
- **Training Mode**: Automatically disabled during inference
- **Random Masking**: Proper statistical dropout implementation

### Activation.cpp - Activation Functions
Complete implementation with forward and backward passes:
- **ReLU**: Fast, prevents vanishing gradients
- **Leaky ReLU**: Prevents dead neurons with small negative slope
- **Sigmoid**: Smooth, outputs between 0-1
- **Tanh**: Symmetric, outputs between -1 and 1
- **Softmax**: Probability distribution for classification

### Loss.cpp - Loss Functions
- **Mean Squared Error**: For regression tasks
- **Cross Entropy**: For multi-class classification
- **Binary Cross Entropy**: For binary classification
- **Mean Absolute Error**: Robust to outliers
- **Safe Implementations**: Prevents numerical instability

### Optimizer.cpp - Training Algorithms
- **SGD**: Basic stochastic gradient descent
- **SGD + Momentum**: Accelerated convergence
- **Adam**: Adaptive learning rates with momentum
- **RMSprop**: Adaptive learning rates
- **Parameter Management**: Automatic state tracking per layer

### Initializer_Weights_Biases.cpp - Weight Initialization
- **Xavier/Glorot**: For sigmoid/tanh activations
- **He Initialization**: For ReLU activations
- **Automatic Selection**: Chooses best method per activation
- **Statistical Properties**: Maintains proper variance scaling

## → Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Guidelines

- Follow C++ best practices and modern C++ standards
- Use descriptive variable and function names
- Add comments for complex algorithms
- Ensure code compiles without warnings
- Test your changes with MNIST dataset

## → Contact

- **GitHub**: [Dora-The-Explorer-Is-Dead](https://github.com/Dora-The-Explorer-Is-Dead)
- **Project Link**: [Neural_Network_Library_CPP](https://github.com/Dora-The-Explorer-Is-Dead/Neural_Network_Library_CPP)

---

**Happy Neural Networking!**
