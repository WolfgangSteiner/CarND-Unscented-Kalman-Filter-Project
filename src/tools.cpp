//==============================================================================================
#include <iostream>
#include <cmath>
#include "tools.h"
//==============================================================================================
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::VectorXd;
using Eigen::MatrixXd;
//==============================================================================================

Eigen::VectorXd Tools::CalculateRMSE(
  const std::vector<VectorXd>& estimations,
  const std::vector<VectorXd>& ground_truth)
{
  assert(estimations.size() > 0);
  assert(estimations.size() == ground_truth.size());

  VectorXd rmse = VectorXd::Zero(estimations[0].rows());

  for (int i = 0; i < estimations.size(); ++i)
  {
    const VectorXd& e = estimations[i];
    const VectorXd& g = ground_truth[i];
    VectorXd r = e-g;
    r = r.array() * r.array();
    rmse += r;
  }
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

//----------------------------------------------------------------------------------------------------------------------

Eigen::MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
  MatrixXd Hj(3,4);
  //recover state parameters
  const float px = x_state(0);
  const float py = x_state(1);
  const float vx = x_state(2);
  const float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  const float c1 = px*px+py*py;
  const float c2 = sqrt(c1);
  const float c3 = (c1*c2);

  //check division by zero
  if(fabs(c1) < 0.0001)
  {
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}

//----------------------------------------------------------------------------------------------------------------------

Eigen::Vector2f Tools::PolarToCartesian(double rho, double phi)
{
  auto result = Vector2f();
  result << rho * cos(phi), rho * sin(phi);
  return result;
}

//==============================================================================================
