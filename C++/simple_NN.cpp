#include "simple_NN.h"


/**
 * class Dense Layer implementation
 */

DenseLayer::DenseLayer(int input_size, int output_size){
    this->input_size = input_size;
    this->output_size = output_size;

    std::vector<double> this->input(input_size);
}

void DenseLayer::forward(std::vector<double> input){
    this->input = input;


}

