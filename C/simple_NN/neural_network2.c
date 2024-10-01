#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.01

enum LayerType {
    denselayer = 0,
    activationlayer = 1,
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
    double* array = (double*) malloc(sizeof(double) * (1+size));
    if (!array) {
        printf("Memory allocation error\n");
        exit(1);
    }
    return array;
}

double** allocate_2d(int rows, int cols) {
    double** array = (double**) malloc((rows+1) * sizeof(double*));
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
    void (*forward)(double* input, struct Layer* layer);
    void (*backward)(double* output_gradient, struct Layer* layer);

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
            //printf("%f\n",layer->output[i]);
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
        //printf("%f \n",layer->activation_layer->activation(input[i]));
        layer->output[i] = layer->activation_layer->activation(input[i]);
    }
}

void backward_dense_layer(double* output_gradient, Layer* layer){

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
    
    printf("%f \n",output_gradient[0]);
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
    Layer* layer = (Layer*) malloc(sizeof(Layer));

    // init sizes
    layer->input_size = input_size;
    layer->output_size = output_size;

    // init input and outputs
    layer->input = allocate_1d(input_size);
    layer->output = allocate_1d(output_size);

    // init forward and backward
    layer->forward = forward_dense_layer;
    layer->backward = backward_dense_layer;

    // set up dense type elements
    layer->type = 0;
    layer->activation_layer = NULL;

    DenseLayer* dense_layer = (DenseLayer*) malloc(sizeof(DenseLayer));

    // init dense layer elements
    dense_layer->biases = allocate_1d(output_size);
    dense_layer->grad_biases = allocate_1d(output_size);
    dense_layer->weights = allocate_2d(output_size,input_size);
    dense_layer->grad_weights = allocate_2d(output_size,input_size);

    layer->dense_layer = dense_layer;

    return layer;
}

Layer* create_activation_layer(int input_size,double (*activation_func)(double), double (*activation_func_prime)(double)){
    Layer* layer = (Layer*) malloc(sizeof(Layer));

    // init sizes
    layer->input_size = input_size;
    layer->output_size = input_size;

    // init input and outputs
    layer->input = allocate_1d(input_size);
    layer->output = allocate_1d(input_size);

    // init forward and backward
    layer->forward = forward_activation_layer;
    layer->backward = backward_activation_layer;

    // set up activation type elements
    layer->type = 1;
    layer->dense_layer = NULL;

    ActivationLayer* activation_layer = (ActivationLayer*) malloc(sizeof(ActivationLayer));

    // init activation layer elements
    activation_layer->activation = activation_func;
    activation_layer->activation_prime = activation_func_prime;

    layer->activation_layer = activation_layer;

    return layer;
}


/**
 * class Network
 */
typedef struct Network {
    Layer** layers;
    int num_layers;
} Network;

Network* create_network(int num_layers) {
    Network* network = (Network*) malloc(sizeof(Network));
    if (!network) {
        printf("Memory allocation error\n");
        exit(1);
    }
    network->num_layers = num_layers;
    network->layers = (Layer**) malloc((num_layers+1) * sizeof(Layer*));
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
    //printf("%f \n", network->layers[0]->output[0]);
}


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
        if (e % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", e, error);
        }
    }
}

// Memory free functions
void free_activation_layer(Layer* layer) {
    free(layer->input);
    free(layer->output);
    free(layer->activation_layer);
}

void free_dense_layer(Layer* layer) {
    for (int i = 0; i < layer->output_size; i++) {
        free(layer->dense_layer->weights[i]);
        free(layer->dense_layer->grad_weights[i]);
    }
    free(layer->dense_layer->weights);
    free(layer->dense_layer->grad_weights);
    free(layer->dense_layer->biases);
    free(layer->dense_layer->grad_biases);
    //free(layer->input);
    free(layer->output);
    free(layer->dense_layer);
    free(layer);
}

int main() {
    // Define the input and output data for XOR problem
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double outputs[4][1] = {{0}, {1}, {1}, {0}};

    // init params
    int num_layer = 4;

    // Convert to double pointers
    double* input_pointers[4];
    double* output_pointers[4];
    for (int i = 0; i < 4; i++) {
        input_pointers[i] = inputs[i];
        output_pointers[i] = outputs[i];
    }

    // Create a network with 1 dense layer and 1 activation layer
    Layer* dense1 = create_dense_layer(2, 4); // 2 inputs, 2 outputs
    Layer* activation1 = create_activation_layer(2, tanh, tanh_prime);
    Layer* dense2 = create_dense_layer(4, 5); // 2 inputs, 1 output
    Layer* activation2 = create_activation_layer(5, tanh, tanh_prime);
    //Layer* dense3 = create_dense_layer(5, 1); // 2 inputs, 1 output
    //Layer* activation3 = create_activation_layer(1, tanh, tanh_prime);

    Network* network = create_network(num_layer);
    network->layers[0] = dense1;
    network->layers[1] = activation1;
    network->layers[2] = dense2;
    network->layers[3] = activation2;
    //network->layers[4] = dense3;
    //network->layers[5] = activation3;

    // Train the network
    train(network, input_pointers, output_pointers, 4, 2, 1, 10000);
    

    // Free memory
    free_dense_layer(dense1);
    free_dense_layer(dense2);
    //free_dense_layer(dense3);
    free_activation_layer(activation1);
    free_activation_layer(activation2);
    //free_activation_layer(activation3);
    free(network->layers);
    free(network);

    return 0;
}





