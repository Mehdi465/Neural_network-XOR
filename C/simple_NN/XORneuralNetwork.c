#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

#define LEARNING_RATE 0.1

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

/* 
 * class BaseLayer
 */

typedef struct Layer  Layer;
typedef struct DenseLayer DenseLayer;
typedef struct ActivationLayer ActivationLayer;

typedef struct Layer{
    double* output;
	int output_size;
	void (*forward)(double* input, Layer* layer);
	void (*backward)(double* input, double* gradient, Layer* layer);
}Layer;

/*
 * Dense class
 */
typedef struct DenseLayer{
	Layer base;
	int input_size;
    double* weights;
	double* biases;
	double* grad_weights;
	double* grad_biases;	
	double* output; // double it bc problem using the one on layer
	int output_size;// double it bc problem using the one on layer
} DenseLayer;

/*
 * Activation class
 */
typedef struct ActivationLayer{
	Layer base;
	double (*activation)(double input);
	double (*activation_prime)(double input);	
	double* output; // double it bc problem using the one on layer
	int output_size;// double it bc problem using the one on layer
} ActivationLayer;

// Memory allocation //
///////////////////////

double* allocate_1D(int size){
	return (double*) malloc(sizeof(double)*size);	
}

/*** DenseLayer definition ***/
/*****************************/

void forward_dense_layer(double* input, Layer* layer){
	DenseLayer* dense_layer = (DenseLayer*)layer;
	
	for(int i = 0; i<dense_layer->base.output_size; i++){
		dense_layer->output[i] = dense_layer->biases[i];
		for(int j = 0; j < dense_layer->input_size; j++){
			dense_layer->output[i] += input[j]*dense_layer->weights[i*dense_layer->output_size + j]; 
		}
	}
}

void backward_dense_layer(double* input, double* gradient,Layer* layer){
	DenseLayer* dense_layer = (DenseLayer*)layer;

	for(int i = 0; i<dense_layer->output_size; i++){
		for(int j = 0; j< dense_layer->input_size; j++){
			dense_layer->grad_weights[i*dense_layer->output_size+j] = input[j]*gradient[i];
			// update weights
			dense_layer->weights[i*dense_layer->output_size+j] -= LEARNING_RATE*(dense_layer->grad_weights[i*dense_layer->output_size+j]);
		}
		dense_layer->grad_biases[i] = gradient[i];
		// update biases
		dense_layer->biases[i] -= LEARNING_RATE*dense_layer->grad_biases[i];
	}
}


//create DenseLayer
DenseLayer* create_dense_layer(int input_size, int output_size){
	DenseLayer* dense_layer = (DenseLayer*) malloc(sizeof(DenseLayer));
	// safe pointer init	
	if(dense_layer == NULL){
		perror("init of denselayer pointer in create_dense_layer failed");
		exit(1);
	}

	// init sizes
	dense_layer->input_size = input_size;
	dense_layer->base.output_size = output_size;
	dense_layer->output_size = output_size;

	srand(time(NULL));
	// init ouput
	dense_layer->base.output = allocate_1D(output_size);
	dense_layer->output = dense_layer->base.output;

	// random init biases
	dense_layer->biases = allocate_1D(output_size);
	for (int i = 0; i < dense_layer->base.output_size; i++){
		dense_layer->biases[i] = ((double)rand()/(double)RAND_MAX); 
	}

	// random init weights
	dense_layer->weights = allocate_1D(input_size*output_size);
	for (int i = 0; i < input_size*output_size; i++){
		dense_layer->weights[i] = ((double)rand()/(double)RAND_MAX); 
	}	

	// init gradient biases and weights
	dense_layer->grad_weights = allocate_1D(input_size*output_size);
	dense_layer->grad_biases = allocate_1D(output_size);


	// link forward and backward functions
	dense_layer->base.forward = forward_dense_layer;
	dense_layer->base.backward = backward_dense_layer;

	return dense_layer;
}


/*** ActivationLayer definition ***/
/**********************************/
void forward_activation_layer(double* input, Layer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
	for (int i = 0; i<activation_layer->base.output_size; i++){
		activation_layer->output[i] = activation_layer->activation(input[i]);	
	}
}

void backward_activation_layer(double* input, double* gradient, Layer* layer){
    ActivationLayer* activation_layer = (ActivationLayer*) layer;
	for (int i = 0; i<activation_layer->output_size; i++){
		gradient[i] *= activation_layer->activation_prime(input[i]);	
	}
}

ActivationLayer* create_activation_layer(int size, 
										double (*activation_func)(double),
										double (*activation_func_prime)(double))
{
	ActivationLayer* activation_layer = (ActivationLayer*) malloc(sizeof(ActivationLayer));
	
	// init size and output
	activation_layer->base.output_size = size;
	activation_layer->output_size = size;
	activation_layer->base.output = allocate_1D(size);
	activation_layer->output = activation_layer->base.output;
	
	// init forward and backward
	activation_layer->base.forward = forward_activation_layer;
	activation_layer->base.backward = backward_activation_layer;
	
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
	for(int i = 0; i < network->num_layer ;i++) {
		// forward for this layer
		network->layers[i]->forward(current_input,network->layers[i]);
		// the output of this layer becomes the input of the next one
		current_input = (double*) network->layers[i]->output;
	}
	// its now the output of the whole network
	return current_input;
}

void network_backward(double* target, Network* network){

	int output_size = network->layers[network->num_layer-1]->output_size;

	double* output_error = (double*)malloc(sizeof(double)*output_size);
	double* current_gradient = NULL;
	int current_gradient_size = 0;

	// init values for the first iteration
	for (int i =0; i<output_size;i++){
		// add here your error metrics (eg mse..)
		output_error[i] = mse_prime(target[i],network->layers[network->num_layer-1]->output[i]);
	}

	current_gradient = output_error;

	for(int i = network->num_layer-1; i>=0; i--){
		output_size = network->layers[i]->output_size;
		network->layers[i]->backward(NULL,current_gradient,network->layers[i]);
    
		if (i!=network->num_layer && current_gradient_size != output_size){
			current_gradient_size = output_size;
			current_gradient = realloc(current_gradient,sizeof(double)*current_gradient_size);
		}
	}
}

Network* create_network(int num_layer){
	Network* network = (Network*) malloc(sizeof(Network));
	network->num_layer = num_layer;
	network->layers = (Layer**) malloc(sizeof(Layer*)*num_layer);

	return network; 
}

void train_network(int num_layer, int epochs, int learning_rate, Network* network,
					double** inputs, double** outputs, int data_size, int num_samples){

	for (int e = 0; e<epochs; e++){
		double error = 0.0;
		for (int i = 0; i<num_samples; i++){
			// forward
			network_forward(inputs[i],network);

			// compute error and gradient error
			double output_size = network->layers[network->num_layer-1]->output_size;
			for(int j = 0; j<output_size ; j++){
				error += mse(outputs[i][j],network->layers[network->num_layer-1]->output[j]);
			}
		
			// backward
			network_backward(inputs[i],network);	
		}

		if (e%200==0){
			printf("epoch n%d error = %f \n",e,error/(num_samples*data_size));
		}
	}
}

int main(){
	// Initialize layers, num_layers, and other parameters

// init of size
	int data_size = 2;
	int output_size = 1;
	int num_samples = 4;

	// dense layers
    DenseLayer* dense_layer1 = create_dense_layer(2,4);
	DenseLayer* dense_layer2 = create_dense_layer(3,1);

	// activation layers
	ActivationLayer* activation_layer1 = create_activation_layer(4,tanh,tanh_prime);
	ActivationLayer* activation_layer2 = create_activation_layer(1,tanh,tanh_prime);

	// create network
	int num_layers = 4;
	Network* network = create_network(num_layers);
	network->layers[0] = dense_layer1;
	network->layers[1] = activation_layer1;
	network->layers[2] = dense_layer2;
	network->layers[3] = activation_layer2;


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

    

	// Cleanup pointers
	free(dense_layer1);
	free(dense_layer2);
	free(activation_layer1);
	free(activation_layer2);

	free(network);

    return 0;
}


