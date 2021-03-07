#include <iostream>
#include "deep_learning/Sigmoid.h"
using namespace std;

int main(){
    xt::xarray<double> arr1
  {{1.0, 2.0, 3.0},
   {2.0, 5.0, 7.0},
   {2.0, 5.0, 7.0}};
   Sigmoid sig = Sigmoid(); 
   xt::xarray<double> a = sig.sigmoid(arr1);
   cout << a ;

    return 0;
}