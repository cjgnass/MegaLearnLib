#include <iostream> 
#include <random> 
#include <Eigen/Dense> 
#include "./algorithms/linear_regression.cpp"


int main() { 

  Eigen::MatrixXd xs(10, 1);
  Eigen::VectorXd ys(10);

  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1, 1);

  for (int i = 0; i < 10; i++) { 
    double r = dist(gen);
    xs(i, 0) = i; 
    ys(i) = (double) 3 * i + 5 + r; 
  } 

  LinearRegression model = LinearRegression(); 
  model.train(xs, ys);

  std::cout << model.getWeights() << std::endl;
  
  
}
