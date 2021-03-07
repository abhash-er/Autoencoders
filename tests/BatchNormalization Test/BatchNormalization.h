#ifndef BATCHNORM_H
#define BATCHNORM_H

#include <iostream>
#include "xtensor/xarray.hpp"
#include "StochasticGradientDescent.h"
#include "Adam.h"
#include "RMSprop.h"
#include "Adadelta.h"


class BatchNormalization{

    //Parameter -> takes in the name of the function

public:
    bool isTrainable;
    double momentum;
    double eps;
    double running_mean;
    double running_var;
    std::vector<int> input_shape;
    xt::xarray<double> gamma;
    xt::xarray<double> beta;
    StochasticGradientDescent sgd;
    Adam adam;
    RMSprop rmsprop;
    Adadelta adadelta;

    
    BatchNormalization(double mom);
    void initialize(std::string optimizer_name,StochasticGradientDescent opt_sgd, Adam opt_adam, RMSprop opt_rms_prop, Adadelta opt_ada);
    int parameters();  
    void set_input_shape(std::vector<int> shape);
    xt::xarray<double> forward_pass(xt::xarray<double> X, bool training = true);
    xt::xarray<double> backward_pass(xt::xarray<double> accum_grad); 
    xt::xarray<double> output_shape();
    

};

#endif 