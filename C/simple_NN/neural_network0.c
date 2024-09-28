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

    void (*forward)(double* input, BaseLayer* layer);
	void (*backward)(double* output_gradient, BaseLayer* layer);
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

void forward_dense_layer(double* input, BaseLayer* layer){
    DenseLayer* dense_layer = (DenseLayer*)layer;
    dense_layer->base->input = input;

    for(int i = 0; i<dense_layer->base->output_size; i++){
		dense_layer->base->output[i] = dense_layer->biases[i];
		for(int j = 0; j < dense_layer->base->input_size; j++){
			dense_layer->base->output[i] += input[j]*dense_layer->weights[i][j]; 
		}
	}
	// realloc size to use it to receive the returned value
	input = realloc(input, dense_layer->base->output_size*sizeof(double));
	// safety mechanisms
	if (input == NULL){
		perror("error in realloc forward dense");
		exit(-1);
	}
	// the input becomes the output (input for the next layer)
	input = dense_layer->base->output;
}

void backward_dense_layer(double* output_gradient, BaseLayer* layer){
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

	output_gradient = realloc(output_gradient,dense_layer->base->input_size*sizeof(double));
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

	dense_layer->base = (BaseLayer*) malloc(sizeof(BaseLayer));

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
	input = realloc(input,activation_layer->base->output_size*sizeof(double));
}

void backward_activation_layer(double* gradient, BaseLayer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
    double* res_grad = allocate_1d(activation_layer->base->input_size);
	for (int i = 0; i<activation_layer->base->output_size; i++){
		res_grad[i] = activation_layer->activation_prime(activation_layer->base->input[i])*gradient[i];	
	}
	gradient = realloc(gradient, activation_layer->base->input_size*sizeof(double));
	if (gradient == NULL){
		perror("error in realloc gardient in backward_activation");
		exit(-1);
	}
	gradient = res_grad;
}

ActivationLayer* create_activation_layer(int size, 
										double (*activation_func)(double),
										double (*activation_func_prime)(double))
{
	ActivationLayer* activation_layer = (ActivationLayer*) malloc(sizeof(ActivationLayer));
	
	activation_layer->base = (BaseLayer*) malloc(sizeof(BaseLayer));

	// init size and output
	activation_layer->base->output_size = size;
	activation_layer->base->input_size = size;
	activation_layer->base->output = allocate_1d(size);
	activation_layer->base->input = allocate_1d(size);
	
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

void network_forward(Network* network,double* input){
	for (int i = 0; i<network->num_layer; i++){
				BaseLayer* layer = network->layers[i];
        		layer->forward(input,network->layers[i]);
				input = layer->output;
            }
}

void network_backward(Network* network,double* grad_err){
	for (int i = network->num_layer-1; i<=0; i--){
				BaseLayer* layer = network->layers[i];
        		layer->backward(grad_err,network->layers[i]);
            }
}

void train_network(int num_layer, int epochs, int learning_rate, Network* network,
					double** inputs, double** outputs, int data_size, int num_samples){

	for (int e = 0; e<epochs; e++){
		double error = 0.0;
		for (int i = 0; i<num_samples; i++){
			// forward 
            double* output_forward = (double*) malloc(sizeof(double)*data_size);//allocate_1d(data_size);
            output_forward = inputs[i];
			network_forward(network,output_forward);
			
			// compute error and gradient error
			double output_size = network->layers[network->num_layer-1]->output_size;
			// grad error
			double* grad_error = allocate_1d(output_size);
			for(int j = 0; j<output_size ; j++){
				error += mse(outputs[i][j],network->layers[network->num_layer-1]->output[j]);
				grad_error[j] = mse_prime(outputs[i][j],network->layers[network->num_layer-1]->output[j]); 
			}

			// backward
			network_backward(network,grad_error);
			
			// free pointers
			free(output_forward);
			free(grad_error);
		}

		if (e%200==0){
			printf("epoch n%d error = %f \n",e,error/(num_samples*data_size));
		}
	}
}

void free_dense_layer(DenseLayer* layer) {
    // weights
    for (int i = 0; i < layer->base->output_size; i++) {
        free(layer->weights[i]);
        free(layer->grad_weights[i]);
    }
    free(layer->weights);
    free(layer->grad_weights);
    // biases and gradients
    free(layer->biases);
    free(layer->grad_biases);
    //input and output
    free(layer->base->input);
    free(layer->base->output);
    // baselayr
    free(layer->base);
    //layer
    free(layer);
}

void free_activation_layer(ActivationLayer* layer) {
    free(layer->base->input);
    free(layer->base->output);
    free(layer->base);
    free(layer);
}

int main(int argc, char* argv[]){
	// Initialize layers, num_layers, and other parameters
	// init of size
	int data_size = 2;
	int output_size = 1;
	int num_samples = 4;
	
	// dense layers
    DenseLayer* dense_layer1 = create_dense_layer(data_size,output_size);
	//DenseLayer* dense_layer2 = create_dense_layer(3,1);

	// activation layers
	//ActivationLayer* activation_layer1 = create_activation_layer(4,tanh,tanh_prime);
	//ActivationLayer* activation_layer2 = create_activation_layer(1,tanh,tanh_prime);

	// create network
	int num_layers = 1;
	Network* network = create_network(num_layers);
	network->layers[0] = dense_layer1;
	/*network->layers[1] = activation_layer1;
	network->layers[2] = dense_layer2;
	network->layers[3] = activation_layer2;*/


    // Example input and output data
    double inputs[4][2] = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    double outputs[4][1] = { {0}, {1}, {1}, {0} };
    
    // Convert to double pointers for the function call
    double* input_pointers[4];
    double* output_pointers[4];
    for (int i = 0; i < 4; i++) {
        input_pointers[i] = inputs[i];
        output_pointers[i] = outputs[i];
    }

    // Train the network
    train_network( num_layers, 10000, 0.01, network, input_pointers, output_pointers, num_samples, data_size);

    
	free_dense_layer(dense_layer1);
	// Cleanup pointers
	free(dense_layer1);
	/*free(dense_layer2);
	free(activation_layer1);
	free(activation_layer2);*/

	free(network);

    return 0;
}





