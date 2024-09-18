#include <stdio.h>
#include <time.h>
#include <math.h>

// activation functions
double sigmoid(double x){
	return 1/(1+exp(-x));	
}

double sigmoid_prime(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

double tanh(double x){
	return 0.0;
}

/* 
 * class BaseLayer
 */
typedef struct Layer{
	void (*forward)(double* input, Layer* layer);
	void (*backward)(double* input, double* gradient, Layer* layer);
}Layer;

/*
 * Dense class
 */
typedef struct DenseLayer{
	Layer base;
	int input_size;
	int output_size;
	double* ouput;
    double* weights;
	double* biases;
	double* grad_weights;
	double* grad_biases;	
} DenseLayer;

/*
 * Activation class
 */
typedef struct ActivationLayer{
	Layer base;
	void (*activaton)(double* input);
	void (*activation_prime)(double* input);	
    double* output;
    int size;
} ActivationLayer;

// Memory allocation //
///////////////////////

double* allocate_1D(int size){
	return (double*) malloc(sizeof(double)*size)	
}

double* allocate_2D(int size_row, int size_column){
	double ** res =(double**) malloc(size(double*)*size_row);
	for (int i = 0; i< size_row; i++){
		res[i] =(double*) malloc(sizeof(double)*size_column);
	}
	return res;
}

/*** DenseLayer definition ***/
/*****************************/

void forward_dense_layer(double* input, Layer* layer){
	DenseLayer* dense_layer = (DenseLayer*)layer;
	
	for(int i = 0; i<dense_layer->output_size; i++){
		dense_layer->output[i] = dense_layer->biases[i];
		for(int j = 0; j < dense_layer->input_size; j++){
			dense_layer->output[i] += input[j]*dense_layer->weights[i*dense_layer->output_size + j]; 
		}
	}
}

void backward_dense_layer(double* input, double* gradient,Layer* layer){
	DenseLayer dense_layer = (DenseLayer*)layer;

	for(int i = 0; i<dense_layer->output_size; i++){
		for(int j = 0; j< dense_layer->input_size; j++){
			dense_layer->grad_weights[i*dense_layer->output_size+j] = input[j]*gradient[i];
		}
		dense_layer->grad_biases[i] = (gradient[i]);
	}
}


//create DenseLayer
DenseLayer* create_dense_layer(int input_size, int output_size){
	DenseLayer* dense_layer = (DenseLayer*) malloc(sizeof(DenseLayer));
	// safe pointer init	
	if(dense_layer = NULL){
		perror("init of denselayer pointer in create_dense_layer failed");
		exit(1);
	}

	// init sizes
	dense_layer->input_size = input_size;
	dense_layer->output_size = output_size;

	srand(NULL);
	// init ouput
	dense_layer->output = allocate_1D(output_size);

	// random init biases
	dense_layer->biases = allocate_1D(output_size);
	for (int i = 0; i < dense_layer->output_size; i++){
		dense_layer->biases[i] = ((double)rand()/(double)RAND_MAX); 

	// random init weights
	dense_layer->weights = allocate_1D(input_size*output_size);
	for (int i = 0; i < input_size*output_size; i++){
		dense_layer->weights[i] = ((double)rand()/(double)RAND_MAX); 

	// init gradient biases and weights
	dense_layer->grad_weights = allocate_1D(input_size*output_size);
	dense_layer->grad_biases = allocate_1D(output_size);

	// link forward and backward functions
	dense_layer->base.foward = forward_dense_layer;
	dense_layer->base.backward = backward_dense_layer;

	return dense_layer
}


/*** ActivationLayer definition ***/
/**********************************/
void forward_activation_layer(double* input, Layer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
	for (int i = 0; i<activation_layer->size; i++){
		activation_layer->output[i] = activation_layer->activation_func(input[i]);	
	}
}

void backward_activation_layer(double* input, double* gradient, Layer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
	for (int i = 0; i<activation_layer->size; i++){
		gradient[i] *= activation_layer->activation_func_prime(input[i]);	
}

ActivationLayer* create_activation_layer(int size, 
										double (*activation_func)(double),
										double (*activation_func_prime)(double))
{
	ActivationLayer* activation_layer = (ActivationLayer*) malloc(sizeof(ActivationLayer));
	
	// init size and output
	activation_layer->size = size;
	activation_layer->output = allocate_1D(size);
	
	// init forward and backward
	activation_layer->base.forward = forward_activation_func;
	activation_layre->base.backward = backward_activation_func;
	
	// init activation, and activation prime
	activation_layer->activation = activation_func;
	activation_layer->activation_prime = activation_func_prime;	

	return activation_layer;
}

/*********************/
// NETWORK STRUCTURE //
/*********************/

/*
 * Network class
 */
typedef struct Network {
	Layer** layers;
	int num_layer;
} Network;

double* network_forward(double* input, Network* network){
	//input will change each iteration in each layer
	double* current_input = input;
	for(int i = 0; i < network->num_layer < i++) {
		// forward for this layer
		network->layers[i]->forward(current_input,network->layer[i]);
		// the output of this layer becomes the input of the next one
		current_input = (double*) network->layers[i]->output;
	}
	// its now the output of the whole network
	return current_input
}

void network_backward(double* gradient, Network* network){
	for(int i = 0; i<network->size; i++){
		network->layers[i]->backward(NULL,gradient,network->layer[i]);
	}
}

Network* create_network(int num_layer){
	Network* network = (Network*) malloc(sizeof(Network));
	network->num_layer = num_layer;
	network->layers = (Layer**) malloc(sizeof(Layer*)*num_layer);

	return network; 
}

void train_network(int num_layer, int epoch, int learning_rate, Layer* layer){
	
}


