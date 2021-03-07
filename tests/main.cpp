#include <iostream>
#include "CrossEntropy.h"
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

  CrossEntropy closs = CrossEntropy();  
  xt::xarray<double> a = closs.loss(arr1,arr2);
  std::cout << a << std::endl; 
  xt::xarray<double> b = closs.gradient(arr1,arr1);
  std::cout << b << std::endl; 
  xt::xarray<double> c = closs.acc(arr1,arr11);
  std::cout << c << std::endl; 

  return 0;
}