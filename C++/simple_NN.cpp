#include "simple_NN.h"


/**
 * class Dense Layer implementation
 */

DenseLayer::DenseLayer() = default;

DenseLayer::DenseLayer(int input_size, int output_size){
    this->input_size = input_size;
    this->output_size = output_size;

    input.resize(input_size);               
    output.resize(output_size);         

    biases.resize(output_size);             
    grad_biases.resize(output_size);        

    weights.resize(output_size, std::vector<double>(input_size));        
    grad_weights.resize(output_size, std::vector<double>(input_size)); 
}

void DenseLayer::forward(std::vector<double> input){
    this->input = input;

}

void DenseLayer::backward(std::vector<double> output_gradient){

}

/**
 * class Activation Layer implementation
 */

ActivationLayer::ActivationLayer() = default;

ActivationLayer::ActivationLayer(int size){};

void ActivationLayer::forward(std::vector<double> input){

}

void ActivationLayer::backward(std::vector<double> output_gradient) {

}

/**
 * class Network implementation
 */

Network::Network() = default;

Network::Network(int num_layers){
    this->num_layer = num_layer;
}

void Network::forward_network(){

}

void Network::backward_network(){

}

void Network::train_network(std::vector<std::vector<double>> inputs,std::vector<std::vector<double>> outputs,
        int num_samples, int input_size,int output_size, int epochs){

}

// MAIN //
//////////

int main(){

}

