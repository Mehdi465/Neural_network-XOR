#include <iostream>
#include <cmath>
#include <vector>
#include <functional>

// learning rate, often called alpha
#define LEARNING_RATE 0.01

enum ActivationFunction {
    Sigmoid,
    ReLu,
    Tanh,
};

class BaseLayer {

    public:

        int input_size;
        int output_size;
        std::vector<double> input;
        std::vector<double> output;

        virtual void forward(std::vector<double> input)=0;
        virtual void backward(std::vector<double> output_gradient)=0;

};

class DenseLayer : public BaseLayer {
    private:
    
        std::vector<double> biases;
        std::vector<double> grad_biases;
        std::vector<std::vector<double>> weights;
        std::vector<std::vector<double>> grad_weights;

    public:

        //Constructors
        DenseLayer();
        DenseLayer(int input_size, int output_size);

        void forward(std::vector<double> input);
        void backward(std::vector<double> output_gradient);

};

class ActivationLayer : public BaseLayer {
    
    public:
        std::function<double(std::vector<double>,std::vector<double>)> activation;
        std::function<double(std::vector<double>,std::vector<double>)> activation_prime;

        // Constructors
        ActivationLayer();
        ActivationLayer(int size);

        void forward(std::vector<double> input);
        void backward(std::vector<double> output_gradient);
};

class Network {
    

    public:

        int num_layer;
        std::vector<BaseLayer> layers;
        
        // Constructors
        Network();
        Network(int num_layer);

        void forward_network();
        void backward_network();
        void train_network(std::vector<std::vector<double>> inputs,std::vector<std::vector<double>> outputs, int num_samples, int input_size,int output_size, int epochs);
};