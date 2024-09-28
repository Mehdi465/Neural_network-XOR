#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum LayerType {
    DenseLayer = 0,
    ActivationLayer = 1,
};


#define LEARNING_RATE 0.01

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prime(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double tanh_activation(double x){
    return tanh(x);
}

double tanh_prime(double x) {
    return 1 - tanh(x) * tanh(x);
}

// Loss function and its derivative (Mean Squared Error)
double mse(double y_true, double y_pred) {
    return (y_true - y_pred) * (y_true - y_pred);
}

double mse_prime(double y_true, double y_pred) {
    return 2 * (y_pred - y_true);
}

// Memory allocation functions with error handling
double* allocate_1d(int size) {
    double* array = (double*) malloc(sizeof(double) * size);
    if (!array) {
        printf("Memory allocation error\n");
        exit(1);
    }
    return array;
}

double** allocate_2d(int rows, int cols) {
    double** array = (double**) malloc(rows * sizeof(double*));
    if (!array) {
        printf("Memory allocation error\n");
        exit(1);
    }

    for (int i = 0; i < rows; i++) {
        array[i] = allocate_1d(cols);
    }
    return array;
}

/*
* class DenseLayer 
*/
typedef struct DenseLayer {
    double** weights;
    double* biases;
    double** grad_weights;
    double* grad_biases;
} DenseLayer;

/*
* class ActivationLayer 
*/ 
typedef struct ActivationLayer {
    BaseLayer base;
    double (*activation)(double x);
    double (*activation_prime)(double x);
} ActivationLayer;

/*
* class BaseLayer 
*/ 
typedef struct BaseLayer {
    int input_size;
    int output_size;
    double* input;
    double* output;
    void (*forward)(double* input, struct BaseLayer* layer);
    void (*backward)(double* output_gradient, struct BaseLayer* layer);
    // enum type
    enum LayerType type;
    ActivationLayer* activation_layer;
    DenseLayer dense_layer;
} BaseLayer;