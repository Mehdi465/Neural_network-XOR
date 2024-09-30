#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.01

enum LayerType {
    DenseLayer = 0,
    ActivationLayer = 1,
};

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
    double (*activation)(double x);
    double (*activation_prime)(double x);
} ActivationLayer;

/*
* class Layer 
*/ 
typedef struct Layer {
    int input_size;
    int output_size;
    double* input;
    double* output;
    void (*forward)(double* input, struct BaseLayer* layer);
    void (*backward)(double* output_gradient, struct BaseLayer* layer);

    enum LayerType type;
    ActivationLayer* activation_layer;
    DenseLayer* dense_layer;
} Layer;

void forward_dense_layer(double* input, Layer* layer){
    // check if its layer type
    if (layer->type != 0){
        perror("wrong layer type");
        exit(-1);
    }

    layer->input = input;
    // Compute the weighted sum
    for (int i = 0; i < layer->output_size; i++) {
        // add biase
        layer->output[i] = layer->dense_layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            layer->output[i] += input[j]*layer->dense_layer->weights[i][j];
        }
    }
}

void forward_activation_layer(double* input, Layer* layer){
    if (layer->type != 1){
        perror("wrong layer type");
        exit(-1);
    }
    layer->input = input;

    for (int i = 0; i < layer->output_size; i++) {
        layer->output[i] = layer->activation_layer->activation(input[i]);
    }
}

void backard_dense_layer(double* output_gradient, Layer* layer){

    if (layer->type != 0){
        perror("wrong layer type");
        exit(-1);
    }

    double* input_gradient = allocate_1d(layer->input_size);

    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            layer->dense_layer->grad_weights[i][j] = layer->input[j] * output_gradient[i];
            input_gradient[j] += layer->dense_layer->weights[i][j] * output_gradient[i];
            // Update the weights
            layer->dense_layer->weights[i][j] -= LEARNING_RATE * layer->dense_layer->grad_weights[i][j];
        }
        // Update biases
        layer->dense_layer->grad_biases[i] = output_gradient[i];
        layer->dense_layer->biases[i] -= LEARNING_RATE * layer->dense_layer->grad_biases[i];
    }

    // Update the gradient for the next layer
    for (int i = 0; i < layer->input_size; i++) {
        output_gradient[i] = input_gradient[i];
    }
    free(input_gradient);
}

void backward_activation_layer(double* output_gradient, Layer* layer){
    if (layer->type != 1){
        perror("wrong layer type");
        exit(-1);
    }
    
    for (int i = 0; i < layer->output_size; i++) {
        output_gradient[i] = output_gradient[i] * layer->activation_layer->activation_prime(layer->input[i]);
    }
}

Layer* create_dense_layer(int input_size, int output_size){
    Layer* dense_layer = (Layer*) malloc(sizeof(Layer));

    // init sizes
    dense_layer->input_size = input_size;
    dense_layer->output_size = output_size;

    // init input and outputs
    dense_layer->input = allocate_1d(input_size);
    dense_layer->output = allocate_1d(output_size);

    // init forward and backward
    dense_layer->forward = forward_dense_layer;
    dense_layer->backward = backard_dense_layer;

    dense_layer->type = 0;
    dense_layer->activation_layer = NULL;

    


    return dense_layer;
}

