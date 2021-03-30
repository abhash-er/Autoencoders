#ifndef DENSE_H
#define DENSE_H

#include <iostream>
#include "xtensor/xarray.hpp"
#include "StochasticGradientDescent.h"
#include "Adam.h"
#include "RMSprop.h"
#include "Adadelta.h"



class Dense{

    //Parameter -> takes in the name of the function

public:
    std::vector<int> input_shape;
    int n_units;
    bool isTrainable;
    xt::xarray<double> layer_input;
    xt::xarray<double> W;
    xt::xarray<double> wo;
    
    std::string opt_name;
    
    StochasticGradientDescent W_opt_sgd;
    Adam W_opt_adam;
    RMSprop W_opt_rms;
    Adadelta W_opt_ada;

    StochasticGradientDescent wo_opt_sgd;
    Adam wo_opt_adam;
    RMSprop wo_opt_rms;
    Adadelta wo_opt_ada;

    Dense(int units, vector<int> shape);

    void intialize(std::string optimizer_name, StochasticGradientDescent opt_sgd, Adam opt_adam, RMSprop opt_rms_prop, Adadelta opt_ada);

    int parameters();

    xt::xarray<double> forward_pass(xt::xarray<double> X, bool training);
    xt::xarray<double> backward_pass(xt::xarray<double> accum_grad);

    vector<int> output_shape();
    

};

#endif 