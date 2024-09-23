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
	void (*backward)(double* input, double* gradient, BaseLayer* layer);
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

    // init weights
    dense_layer->weights = allocate_2d(output_size,input_size); 


    

	

	return dense_layer;
}

