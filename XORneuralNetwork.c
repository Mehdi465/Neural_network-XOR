#include <stdio.h>
#include <time.h>
#include <math.h>

#define LEARNING_RATE 0.1

// error functions
double mse(double y_true, double y_pred){
	return 
}

double mse_prime(){
}

// activation functions
double sigmoid(double x){
	return 1/(1+exp(-x));	
}

double sigmoid_prime(double x){
	return sigmoid(x)*(1-sigmoid(x));
}

double tanh(double x){
	return 
}

/* 
 * class BaseLayer
 */
struct Layer{
	void (*forward)(double* input, Layer* layer);
	void (*backward)(double* input, double* gradient, Layer* layer);
};

/*
 * Dense class
 */
typedef struct DenseLayer{
	struct Layer base;
	int input_size;
	int output_size;
	double* ouput;
    double* weights;
	double* biases;
	double* grad_weights;
	double* grad_biases;
	
	Dense* (*create_dense_layer)(int input_size, int output_size);
} Dense;

/*
 * Activation class
 */

typedef struct ActivationLayer{
	struct Layer base;
	void (*activaton)(double input);
	void (*activation_prime)(double* input);	
    double* output;
} ActivationLayer;



void forward_dense_layer(double* input, struct Layer* layer){
	struct DenseLayer* dense_layer = (struct DenseLayer*)layer;
	
	for(int i = 0; i<dense_layer->output_size; i++){
		dense_layer->output[i] = dense_layer->biases[i];
		for(int j = 0; j < dense_layer->input_size; j++){
			dense_layer->output[i] += input[j]*dense_layer->weights[i*dense_layer->output_size + j]; 
		}
	}
}

void backward_dense_layer(double* input, double* gradient,struct Layer* layer){
	struct DenseLayer dense_layer = (struct DenseLayer*)layer;

	for(int i = 0; i<dense_layer->output_size; i++){
		for(int j = 0; j< dense_layer->input_size; j++){
			dense_layer->grad_weights[i*dense_layer->output_size+j] = (1-LEARNING_RATE)*input[j]*gradient[i];
		}
		dense_layer->grad_biases[i] = (1-LEARNING_RATE)*(gradient[i]);
	}
}


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

//create DenseLayer
struct DenseLayer* create_dense_layer(int input_size, int output_size){
	struct DenseLayer dense_layer = (struct DenseLayer*) malloc(sizeof(struct DenseLayer));

	dense_layer->input_size = input_size;
	dense_layer->output_size = output_size;

	srand(NULL);
	// random init biases
	dense_layer->biases = allocate_1D(output_size);	

	// random init weights
	dense_layer->weights = allocate_1D(input_size*output_size);
	
	
}
