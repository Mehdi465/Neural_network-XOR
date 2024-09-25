#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define LEARNING_RATE 0.1

typedef struct BaseLayer BaseLayer;
typedef struct DenseLayer DenseLayer;
typedef struct ActivationLayer ActivationLayer;


// activation functions
double sigmoid(double x){
	return 1/(1+exp(-x));	
}

double sigmoid_prime(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

double tanh_prime(double x){
	return 1-tanh(x)*tanh(x);
}

double mse(double y, double y_pred){
	return (y-y_pred)*(y-y_pred);
}

double mse_prime(double y, double y_pred){
	return 2*(y-y_pred);
}

// Memory allocation //
///////////////////////

double* allocate_1d(int size){
	return (double*) malloc(sizeof(double)*size);	
}

double** allocate_2d(int rows, int cols) {
    double** array = (double**)malloc(sizeof(double*) * rows);
    for (int i = 0; i < rows; i++) {
        array[i] = allocate_1d(cols);
    }
    return array;
}

/* 
 * class BaseLayer
 */

struct BaseLayer {
    int input_size;
    int output_size;
    double* input;
    double* output;

    void* (*forward)(double* input, BaseLayer* layer);
	void* (*backward)(double* output_gradient, BaseLayer* layer);
};

/* 
 * class DenseLayer
 */

struct DenseLayer{
    BaseLayer* base;
    double** weights;
	double* biases;
	double** grad_weights;
	double* grad_biases;
};

void* forward_dense_layer(double* input, DenseLayer* layer){
    DenseLayer* dense_layer = (DenseLayer*)layer;
    dense_layer->base->input = input;
    double* output = allocate_1d(dense_layer->base->output_size);

    for(int i = 0; i<dense_layer->base->output_size; i++){
		dense_layer->base->output[i] = dense_layer->biases[i];
		for(int j = 0; j < dense_layer->base->input_size; j++){
			dense_layer->base->output[i] += input[j]*dense_layer->weights[i][j]; 
		}
	}
	// realloc size to use it to receive the returned value
	input = realloc(input, dense_layer->base->output_size);
	// safety mechanisms
	if (input == NULL){
		perror("error in realloc forward dense");
		exit(-1);
	}
	// the input becomes the output (input for the next layer)
	input = output
}

void backward_dense_layer(double* output_gradient, DenseLayer* layer){
    DenseLayer* dense_layer = (DenseLayer*)layer;
    
    double* res_gradient = allocate_1d(dense_layer->base->input_size);

    for(int i = 0; i<dense_layer->base->output_size; i++){
		for(int j = 0; j< dense_layer->base->input_size; j++){
			dense_layer->grad_weights[i][j] = dense_layer->base->input[j]*output_gradient[i];
            // update weights
            dense_layer->weights[i][j] -= LEARNING_RATE*(dense_layer->grad_weights[i][j]);
            res_gradient[j] += output_gradient[i]*dense_layer->weights[i][j];
        }
        dense_layer->biases[i] -= LEARNING_RATE*output_gradient[i];
    }

	output_gradient = realloc(output_gradient,layer->base->input_size);
	if (output_gradient == NULL){
		perror("error realloc in backward dense");
		exit(-1);
	}
	output_gradient = res_gradient;
}

//create DenseLayer
DenseLayer* create_dense_layer(int input_size, int output_size){
	DenseLayer* dense_layer = (DenseLayer*) malloc(sizeof(DenseLayer));
	// safe pointer init	
	if(dense_layer == NULL){
		perror("init of denselayer pointer in create_dense_layer failed");
		exit(1);
	}
    // init base size
    dense_layer->base->input_size = input_size;
    dense_layer->base->output_size = output_size;

    // init base input and output
    dense_layer->base->input = allocate_1d(input_size);
    dense_layer->base->output = allocate_1d(output_size);

    // init biases and grad biases
    dense_layer->biases = allocate_1d(output_size);
    for (int i = 0; i < dense_layer->base->output_size; i++){
        // random init
    	dense_layer->biases[i] = ((double)rand()/(double)RAND_MAX); 
	}
    dense_layer->grad_biases = allocate_1d(output_size);

    // init weights random, and grad_weights
    dense_layer->weights = allocate_2d(output_size,input_size);
    for (int i = 0; i < output_size; i++){
        for(int j = 0; j < input_size; j++){
            dense_layer->weights[i][j] = ((double)rand()/(double)RAND_MAX); 
        }
	}	
    dense_layer->grad_weights = allocate_2d(output_size,input_size); 

    dense_layer->base->forward = forward_dense_layer;
	dense_layer->base->backward = backward_dense_layer;

	return dense_layer;
}

/*
 * Activation class
 */
typedef struct ActivationLayer{
	BaseLayer* base;

	double (*activation)(double input);
	double (*activation_prime)(double input);	
} ActivationLayer;

void forward_activation_layer(double* input, BaseLayer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
    double* res = allocate_1d(activation_layer->base->input_size);
    activation_layer->base->input = input;
	for (int i = 0; i<activation_layer->base->output_size; i++){
		activation_layer->base->output[i] = activation_layer->activation(input[i]);	
	}
    res = activation_layer->base->output;
	input = realloc(input,layer->base->)
}

void backward_activation_layer(double* gradient, BaseLayer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
    double* res_grad = allocate_1d(activation_layer->base->input_size);
	for (int i = 0; i<activation_layer->base->output_size; i++){
		res_grad[i] = activation_layer->activation_prime(activation_layer->base->input[i])*gradient[i];	
	}
	gradient = realloc(gradient, layer->base->input_size);
	if (gradient == NULL){
		perror("error in realloc gardient in backward_activation");
		exit(-1);
	}
	gradient = res_gradient;
}

ActivationLayer* create_activation_layer(int size, 
										double (*activation_func)(double),
										double (*activation_func_prime)(double))
{
	ActivationLayer* activation_layer = (ActivationLayer*) malloc(sizeof(ActivationLayer));
	
	// init size and output
	activation_layer->base->output_size = size;
	activation_layer->base->input_size = size;
	activation_layer->base->output = allocate_1D(size);
	activation_layer->base->input = allocate_1D(size);
	
	// init forward and backward
	activation_layer->base->forward = forward_activation_layer;
	activation_layer->base->backward = backward_activation_layer;
	
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
	BaseLayer** layers;
	int num_layer;
} Network;

Network* create_network(int num_layer){
	Network* network = (Network*) malloc(sizeof(Network));
	network->num_layer = num_layer;
	network->layers = (BaseLayer**) malloc(sizeof(BaseLayer*)*num_layer);

	return network; 
}

void train_network(int num_layer, int epochs, int learning_rate, Network* network,
					double** inputs, double** outputs, int data_size, int num_samples){

	for (int e = 0; e<epochs; e++){
		double error = 0.0;
		for (int i = 0; i<num_samples; i++){
			// forward 
            double* output_forward = allocate_1d(data_size);
            output_forward = inputs[i];
            for (int j = 0; j<num_layer; j++){
				output_forward = realloc(output_forward, network->layers[j]->base->
                output_forward = network->layers[j]->forward(output_forward,network->layers[j]);
            }
			
			// compute error and gradient error
			double output_size = network->layers[network->num_layer-1]->output_size;
			for(int j = 0; j<output_size ; j++){
				error += mse(outputs[i][j],network->layers[network->num_layer-1]->output[j]);
			}
		
			// backward
			int output_size = network->layers[num_layer-1]->base->output_size;
			double* grad_output = allocate_1d(output_size);
			for(int	j = 0; j < output_size; j++){
				grad_output[j] = mse_prime(inputs[i][j],output_forward[])
			}
		}

		if (e%200==0){
			printf("epoch n%d error = %f \n",e,error/(num_samples*data_size));
		}
	}
}


