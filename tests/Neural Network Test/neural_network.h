#ifndef NEURALNET_H
#define NEURALNET_H

#include <iostream>
#include "xtensor/xarray.hpp"
#include <unordered_map>
#include "StochasticGradientDescent.h"
#include "Adam.h"
#include "RMSprop.h"
#include "Adadelta.h"

#include "Sigmoid.h"
#include "Softmax.h"
#include "Tanh.h"
#include "SoftPlus.h"


#include "Dense.h"
#include "Activation.h"
#include "BatchNormalization.h"
#include "Dropout.h"

class layer_container: public Dense, public Activation, public BatchNormalization, public Dropout{
    layer_container(Dense x){}
    layer_container(Activation y){}
    layer_container(BatchNormalization z){}
    layer_container(Dropout w){}
}

class loss_container:public Sigmoid, public SoftMax, public Tanh, public SoftPlus{
    loss_container(Sigmoid x){}
    loss_container(SoftMax y){}
    loss_container(Tanh z){}
    loss_container(SoftPlus w){}
}

class optimizer_container:public StochasticGradientDescent, public Adam, public RMSprop, public Adadelta{
    optimizer_container(StochasticGradientDescent x){}
    optimizer_container(Adam y){}
    optimizer_container(Adadelta z){}
    optimizer_container(RMSprop w){}
}


class NeuralNetwork{

public:
    optimizer_container optimizer;

    std::vector<layer_container> layers;
    loss_container loss_function;

    unordered_map<std::string, xt::xarray<double>> val_set;
    unordered_map<std::string, std::vector<double>> errors;
    bool isValidationPresent;

    NeuralNetwork();
    void set_trainable();
    void add();
    std::vector<double> test_on_batch(xt::xarray<double> X, xt::xarray<double> y);
    std::vector<double> train_on_batch(xt::xarray<double> X, xt::xarray<double> y);
    std::vector<std::vector<double>> fit(xt::xarray<double> X, xt::xarray<double> y, int n_epochs, int batch_size);
    xt::xarray<double> _forward_pass(xt::xarray<double> X, bool isTrainable);
    void _backward_pass(xt::xarray<double> loss_grad);
    void summary(std::string name);
    xt::xarray<double> predict(xt::xarray<double> X);
};

#endif 