#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

// BaseLayer definition
typedef struct BaseLayer {
    int input_size;
    int output_size;
    double* input;
    double* output;
    void (*forward)(double* input, struct BaseLayer* layer);
    void (*backward)(double* output_gradient, struct BaseLayer* layer);
} BaseLayer;

// DenseLayer definition
typedef struct DenseLayer {
    BaseLayer base;
    double** weights;
    double* biases;
    double** grad_weights;
    double* grad_biases;
} DenseLayer;

void free_dense_layer(DenseLayer* layer) {
    for (int i = 0; i < layer->base.output_size; i++) {
        free(layer->weights[i]);
        free(layer->grad_weights[i]);
    }
    free(layer->weights);
    free(layer->grad_weights);
    free(layer->biases);
    free(layer->grad_biases);
    //free(layer->base.input);
    //free(layer->base.output);
    free(layer);
}

void forward_dense_layer(double* input, BaseLayer* layer) {
    DenseLayer* dense = (DenseLayer*) layer;
    dense->base.input = input;

    // Compute the weighted sum
    for (int i = 0; i < dense->base.output_size; i++) {
        // add biase
        dense->base.output[i] = dense->biases[i];
        for (int j = 0; j < dense->base.input_size; j++) {
            dense->base.output[i] += input[j]*dense->weights[i][j];
        }
    }
}

void backward_dense_layer(double* output_gradient, BaseLayer* layer) {
    DenseLayer* dense = (DenseLayer*) layer;
    double* input_gradient = allocate_1d(dense->base.input_size);

    // Calculate weight gradients and propagate back to previous layer
    for (int i = 0; i < dense->base.output_size; i++) {
        for (int j = 0; j < dense->base.input_size; j++) {
            dense->grad_weights[i][j] = dense->base.input[j] * output_gradient[i];
            input_gradient[j] += dense->weights[i][j] * output_gradient[i];
            // Update the weights
            dense->weights[i][j] -= LEARNING_RATE * dense->grad_weights[i][j];
        }
        // Update biases
        dense->grad_biases[i] = output_gradient[i];
        dense->biases[i] -= LEARNING_RATE * dense->grad_biases[i];
    }

    // Update the gradient for the next layer
    for (int i = 0; i < dense->base.input_size; i++) {
        output_gradient[i] = input_gradient[i];
    }
    free(input_gradient);
}

DenseLayer* create_dense_layer(int input_size, int output_size) {
    DenseLayer* dense = (DenseLayer*) malloc(sizeof(DenseLayer));
    if (!dense) {
        printf("Memory allocation error\n");
        exit(1);
    }
    dense->base.input_size = input_size;
    dense->base.output_size = output_size;

    // Initialize input and output arrays
    dense->base.input = allocate_1d(input_size);
    dense->base.output = allocate_1d(output_size);

    // Initialize weights, biases, and gradients
    dense->weights = allocate_2d(output_size, input_size);
    dense->biases = allocate_1d(output_size);
    dense->grad_weights = allocate_2d(output_size, input_size);
    dense->grad_biases = allocate_1d(output_size);

    // Random initialization of weights and biases
    for (int i = 0; i < output_size; i++) {
        dense->biases[i] = ((double) rand() / RAND_MAX);
        for (int j = 0; j < input_size; j++) {
            dense->weights[i][j] = ((double) rand() / RAND_MAX);
        }
    }

    // Set forward and backward functions
    dense->base.forward = forward_dense_layer;
    dense->base.backward = backward_dense_layer;

    return dense;
}

// Activation Layer definition
typedef struct ActivationLayer {
    BaseLayer base;
    double (*activation)(double x);
    double (*activation_prime)(double x);
} ActivationLayer;

void forward_activation_layer(double* input, BaseLayer* layer) {
    ActivationLayer* activation = (ActivationLayer*) layer;
    activation->base.input = input;

    for (int i = 0; i < activation->base.output_size; i++) {
        activation->base.output[i] = activation->activation(input[i]);
    }
}

void backward_activation_layer(double* output_gradient, BaseLayer* layer) {
    ActivationLayer* activation = (ActivationLayer*) layer;

    for (int i = 0; i < activation->base.output_size; i++) {
        output_gradient[i] = output_gradient[i] * activation->activation_prime(activation->base.input[i]);
    }
}

ActivationLayer* create_activation_layer(int size, double (*activation_func)(double), double (*activation_func_prime)(double)) {
    ActivationLayer* activation = (ActivationLayer*) malloc(sizeof(ActivationLayer));
    if (!activation) {
        printf("Memory allocation error\n");
        exit(1);
    }
    activation->base.input_size = size;
    activation->base.output_size = size;

    // Initialize input and output arrays
    activation->base.input = allocate_1d(size);
    activation->base.output = allocate_1d(size);

    // Set the activation function and its derivative
    activation->activation = activation_func;
    activation->activation_prime = activation_func_prime;

    // Set forward and backward functions
    activation->base.forward = forward_activation_layer;
    activation->base.backward = backward_activation_layer;

    return activation;
}

// Network definition
typedef struct Network {
    BaseLayer** layers;
    int num_layers;
} Network;

Network* create_network(int num_layers) {
    Network* network = (Network*) malloc(sizeof(Network));
    if (!network) {
        printf("Memory allocation error\n");
        exit(1);
    }
    network->num_layers = num_layers;
    network->layers = (BaseLayer**) malloc(num_layers * sizeof(BaseLayer*));
    if (!network->layers) {
        printf("Memory allocation error\n");
        exit(1);
    }
    return network;
}

void network_forward(Network* network, double* input) {
    for (int i = 0; i < network->num_layers; i++) {
        network->layers[i]->forward(input, network->layers[i]);
        input = network->layers[i]->output;
    }
}

void network_backward(Network* network, double* grad_output) {
    for (int i = network->num_layers - 1; i >= 0; i--) {
        network->layers[i]->backward(grad_output, network->layers[i]);
    }
}

// Training function
void train(Network* network, double** inputs, double** outputs, int num_samples, int input_size, int output_size, int epochs) {
    for (int e = 0; e < epochs; e++) {
        double error = 0;
        for (int i = 0; i < num_samples; i++) {
            // Forward pass
            network_forward(network, inputs[i]);

            // Compute error and gradient for the output layer
            double* output = network->layers[network->num_layers - 1]->output;
            double* grad_output = allocate_1d(output_size);
            for (int j = 0; j < output_size; j++) {
                error += mse(outputs[i][j], output[j]);
                grad_output[j] = mse_prime(outputs[i][j], output[j]);
            }

            // Backward pass
            network_backward(network, grad_output);

            free(grad_output);
        }
        if (e % 100 == 0) {
            printf("Epoch %d, Error: %f\n", e, error);
        }
    }
}

// Memory free functions
void free_activation_layer(ActivationLayer* layer) {
    free(layer->base.input);
    free(layer->base.output);
    free(layer);
}

int main() {
    // Define the input and output data for XOR problem
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double outputs[4][1] = {{0}, {1}, {1}, {0}};

    // init params
    int num_layer = 6;

    // Convert to double pointers
    double* input_pointers[4];
    double* output_pointers[4];
    for (int i = 0; i < 4; i++) {
        input_pointers[i] = inputs[i];
        output_pointers[i] = outputs[i];
    }

    // Create a network with 1 dense layer and 1 activation layer
    DenseLayer* dense1 = create_dense_layer(2, 4); // 2 inputs, 2 outputs
    ActivationLayer* activation1 = create_activation_layer(2, tanh, tanh_prime);
    DenseLayer* dense2 = create_dense_layer(4, 5); // 2 inputs, 1 output
    ActivationLayer* activation2 = create_activation_layer(5, tanh, tanh_prime);
    DenseLayer* dense3 = create_dense_layer(5, 1); // 2 inputs, 1 output
    ActivationLayer* activation3 = create_activation_layer(1, tanh, tanh_prime);

    Network* network = create_network(num_layer);
    network->layers[0] = (BaseLayer*) dense1;
    network->layers[1] = (BaseLayer*) activation1;
    network->layers[2] = (BaseLayer*) dense2;
    network->layers[3] = (BaseLayer*) activation2;
    network->layers[4] = (BaseLayer*) dense3;
    network->layers[5] = (BaseLayer*) activation3;

    // Train the network
    train(network, input_pointers, output_pointers, 4, 2, 1, 10000);

    // Free memory
    free_dense_layer(dense1);
    free_dense_layer(dense2);
    free_dense_layer(dense3);
    free_activation_layer(activation1);
    free_activation_layer(activation2);
    free_activation_layer(activation3);
    free(network->layers);
    free(network);

    return 0;
}
