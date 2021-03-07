#include "BatchNormalization.h"

BatchNormalization::BatchNormalization(double mom){
    momentum = mom;
    eps = 0.01;
    running_mean = 0;
    running_var = 0;
}

void BatchNormalization::initialize(std::string optimizer_name){
    gamma = xt::ones<double>(input_shape);
    beta = xt::zeros<double>(input_shape);
    
}