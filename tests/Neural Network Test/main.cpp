#include <iostream>
#include "Activation.h"
#include <xtensor/xio.hpp>

int main(){
  xt::xarray<double> arr1
{{1.0, 2.0, 3.0},
  {2.0, 5.0, 7.0},
  {2.0, 5.0, 7.0}};

  xt::xarray<double> arr11
{{1.3, 2.8, 3.1},
  {2.0, 5.0, 7.0},
  {2.1, 5.0, 7.0}};


  xt::xarray<double> arr2 {5.0,6.0,7.0};

  Activation active = Activation("sigmoid");
  active.forward_pass(arr1);
  
  return 0;
}