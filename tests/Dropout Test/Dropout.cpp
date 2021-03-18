#include "Dropout.h"

Dropout::Dropout(double p_var = 0.2){
    p = p_var;
}

xt::xarray<double> Dropout::forward_pass(xt::xarray<double> X, bool training = true){
    double c = 1-p;
    if(training){
        //TODO
        //mask 
    }
}