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

/**
 * class Activation Layer implementation
 */

ActivationLayer::ActivationLayer() = default;

ActivationLayer::ActivationLayer(int size){};



int main(){

}

