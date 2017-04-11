//==============================================================================================
#include <iostream>
#include "ukf.h"
#include "tools.h"
//==============================================================================================
using Eigen::VectorXd;
using Eigen::MatrixXd;
//==============================================================================================
static const int n_x = 5;
static const int n_aug = 7;
static const int n_sigma = 2 * n_aug + 1;
static const int n_z_radar = 3;
static const int n_z_laser = 2;
//==============================================================================================

static VectorXd SPrepareWeightVector()
{
  double lambda = 3 - n_aug;
  VectorXd w = 0.5 * VectorXd::Ones(n_sigma);
  w(0) = lambda;
  w /= (lambda + n_aug);
  return w;
}


//----------------------------------------------------------------------------------------------

static double SNormalizeAngle(double phi)
{
  const double Max = M_PI;
  const double Min = -M_PI;

  return phi < Min
    ? Max + std::fmod(phi - Min, Max - Min)
    : std::fmod(phi - Min, Max - Min) + Min;
}


//----------------------------------------------------------------------------------------------

static void SNormalizeAngle(VectorXd& z, int idx)
{
  z(idx) = SNormalizeAngle(z(idx));
}


//----------------------------------------------------------------------------------------------

static VectorXd SCTRVModel(const VectorXd& aX_aug, float aDeltaT)
{
  const double v = aX_aug(2);
  const double psi = aX_aug(3);
  const double psi_dot = aX_aug(4);
  const double nu_a = aX_aug(5);
  const double nu_psi_dot_dot = aX_aug(6);

  const double cos_psi = cos(psi);
  const double sin_psi = sin(psi);
  const double dt2 = aDeltaT * aDeltaT;

  const VectorXd xk = aX_aug.head(5);

  VectorXd v1(5,1), v2(5,1);
  v2 << 0.5 * dt2 * cos_psi * nu_a,
        0.5 * dt2 * sin_psi * nu_a,
        aDeltaT * nu_a,
        0.5 * dt2 * nu_psi_dot_dot,
        aDeltaT * nu_psi_dot_dot;

  if (std::abs(psi_dot) <= 0.001)
  {
    v1 << v * cos_psi * aDeltaT,
          v * sin_psi * aDeltaT,
          0,
          psi_dot * aDeltaT,
          0;
  }
  else
  {
    v1 << v / psi_dot * (sin(psi + psi_dot * aDeltaT) - sin(psi)),
          v / psi_dot * (-cos(psi + psi_dot * aDeltaT) + cos(psi)),
          0,
          psi_dot * aDeltaT,
          0;
  }

  return xk + v1 + v2;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SGenerateSigmaPoints(
  const VectorXd& aX,
  const MatrixXd& aP,
  const VectorXd& aProcessNoise)
{
  //define spreading parameter
  const double lambda = 3 - n_aug;

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug, n_aug);
  P_aug.topLeftCorner(n_x, n_x) = aP;
  P_aug(5, 5) = aProcessNoise(0);
  P_aug(6, 6) = aProcessNoise(1);

  //create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug);
  x_aug.head(n_x) = aX;

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sigma);
  const float f = sqrt(lambda + n_aug);
  const MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug; ++i)
  {
    Xsig_aug.col(i + 1) = x_aug + f * A.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - f * A.col(i);
  }

  return Xsig_aug;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SPredictSigmaPoints(const MatrixXd& aXsig_aug, double aDeltaT)
{
  MatrixXd Xsig_pred = MatrixXd(n_x, n_sigma);

  for (int i = 0; i < n_sigma; ++i)
  {
    Xsig_pred.col(i) = SCTRVModel(aXsig_aug.col(i), aDeltaT);
  }

  return Xsig_pred;
}


//----------------------------------------------------------------------------------------------

static VectorXd SPredictMean(const MatrixXd& aXsig_pred, const VectorXd& aWeights)
{
  VectorXd x = VectorXd::Zero(aXsig_pred.rows());
  for (int i = 0; i < n_sigma; ++i)
  {
    x += aWeights(i) * aXsig_pred.col(i);
  }

  return x;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SPredictCovariance(const MatrixXd& aXsig_pred, const VectorXd& aX, const VectorXd& aWeights)
{
  MatrixXd P = MatrixXd::Zero(aX.rows(), aX.rows());

  for (int i = 0; i < n_sigma; ++i)
  {
    const VectorXd temp = aXsig_pred.col(i) - aX;
    P = P + aWeights(i) * temp * temp.transpose();
  }

  return P;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SCalcCrossCorrelationMatrix(
  const MatrixXd& Xsig, const VectorXd& x,
  const MatrixXd& Zsig, const VectorXd& z,
  const VectorXd& w)
{
  MatrixXd Tc = MatrixXd::Zero(x.rows(), z.rows());

  for (int i = 0; i < n_sigma; ++i)
  {
    VectorXd x_diff = Xsig.col(i) - x;
    VectorXd z_diff = Zsig.col(i) - z;
    SNormalizeAngle(x_diff, 3);
    SNormalizeAngle(z_diff, 1);
    Tc += w(i) * x_diff * z_diff.transpose();
  }

  return Tc;
}


//----------------------------------------------------------------------------------------------

static VectorXd STransformSigmaPointIntoRadarMeasurementSpace(const VectorXd& x)
{
  const double px = x(0);
  const double py = x(1);
  const double v = x(2);
  const double psi = x(3);
  const double rho = sqrt(pow(x(0),2) + pow(x(1),2));
  const double phi = atan2(x(1),x(0));
  const double rho_dot = (px*cos(psi)*v + py*sin(psi)*v) / std::max(rho,1.0e-5);

  VectorXd z_out(3);
  z_out << rho, phi, rho_dot;
  return z_out;
}


//----------------------------------------------------------------------------------------------

static MatrixXd STransformSigmaPointsIntoRadarMeasurementSpace(const MatrixXd& aXsig_pred)
{
  MatrixXd Zsig = MatrixXd(n_z_radar, n_sigma);

  for (int i = 0; i < n_sigma; i++)
  {
    Zsig.col(i) = STransformSigmaPointIntoRadarMeasurementSpace(aXsig_pred.col(i));
  }

  return Zsig;
}


//----------------------------------------------------------------------------------------------

static VectorXd STransformSigmaPointIntoLidarMeasurementSpace(const VectorXd& x)
{
  const double& px = x(0);
  const double& py = x(1);

  VectorXd z_out(2);
  z_out << px, py;
  return z_out;
}


//----------------------------------------------------------------------------------------------

static MatrixXd STransformSigmaPointsIntoLidarMeasurementSpace(const MatrixXd& aXsig_pred)
{
  MatrixXd Zsig = MatrixXd(n_z_laser, n_sigma);

  for (int i = 0; i < n_sigma; i++)
  {
    Zsig.col(i) = STransformSigmaPointIntoLidarMeasurementSpace(aXsig_pred.col(i));
  }

  return Zsig;
}

//==============================================================================================

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
: is_initialized_(false)
, use_laser_(true)
, use_radar_(true)
{
  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(n_x,n_x);
  for (int i = 0; i < n_x; ++i)
  {
      P_(i,i) = 0.01;
  }

  weights_ = SPrepareWeightVector();

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  R_radar_ = MatrixXd::Zero(3,3);
  R_radar_(0,0) = pow(std_radr_,2);
  R_radar_(1,1) = pow(std_radphi_,2);
  R_radar_(2,2) = pow(std_radrd_,2);

  R_lidar_ = MatrixXd::Zero(2,2);
  R_lidar_(0,0) = pow(std_laspx_, 2);
  R_lidar_(1,1) = pow(std_laspy_, 2);

  process_noise_ = VectorXd::Zero(2);
  process_noise_ << pow(std_a_, 2), pow(std_yawdd_, 2);
}


//----------------------------------------------------------------------------------------------
/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& m)
{
  if (!is_initialized_)
  {
    Initialize(m);
    is_initialized_ = true;
    return;
  }

  double delta_t = (m.timestamp_ - previous_timestamp_) / 1.0e6;

  // Dividing large time steps into smaller prediction intervals helps to maintain
  // numerical stability.
  while (delta_t > 0.1)
  {
    Predict(0.05);
    delta_t -= 0.05;
  }

  Predict(delta_t);
  previous_timestamp_ = m.timestamp_;

  if (m.IsLaserMeasurement() && use_laser_)
  {
    UpdateLidar(m);
  }
  else if (m.IsRadarMeasurement() && use_radar_)
  {
    UpdateRadar(m);
  }
}


//----------------------------------------------------------------------------------------------

void UKF::Initialize(const MeasurementPackage& m)
{
  const auto& z = m.Measurement();

  if (m.IsRadarMeasurement())
  {
    const double& rho = z(0);
    const double& phi = z(1);
    const double& rho_dot = z(2);
    const auto p = Tools::PolarToCartesian(rho, phi);
    x_ << p(0), p(1), rho_dot, 0.0, 0.0;
  }
  else if (m.IsLaserMeasurement())
  {
    const double px = z(0);
    const double py = z(1);
    x_ << px, py, 0.0, 0.0, 0.0;
  }

  previous_timestamp_ = m.timestamp_;
}


//----------------------------------------------------------------------------------------------

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */

void UKF::Predict(double aDeltaT)
{
  const MatrixXd Xsig = SGenerateSigmaPoints(x_, P_, process_noise_);

  //std::cout << "Xsig" << std::endl;
  //std::cout << Xsig << std::endl;

  Xsig_pred_ = SPredictSigmaPoints(Xsig, aDeltaT);
  //std::cout << "Xsig_pred_" << std::endl;
  //std::cout << Xsig_pred_ << std::endl;

  x_ = SPredictMean(Xsig_pred_, weights_);
  SNormalizeAngle(x_, 3);
  assert(x_(0) == x_(0));
  assert(x_(1) == x_(1));
  P_ = SPredictCovariance(Xsig_pred_, x_, weights_);
  //P_ = 0.5 * (P_ + P_.transpose());
}


//----------------------------------------------------------------------------------------------

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */

void UKF::UpdateLidar(const MeasurementPackage& m)
{
  const VectorXd& z = m.Measurement();
  const MatrixXd Zsig = STransformSigmaPointsIntoLidarMeasurementSpace(Xsig_pred_);

  // predict radar measurement and covariance
  const VectorXd z_pred = SPredictMean(Zsig, weights_);
  const MatrixXd S = SPredictCovariance(Zsig, z_pred, weights_) + R_lidar_;

  const MatrixXd Tc = SCalcCrossCorrelationMatrix(Xsig_pred_, x_, Zsig, z_pred, weights_);
  const MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;

  x_ += K * z_diff;
  SNormalizeAngle(x_, 3);
  assert(x_(0) == x_(0));
  assert(x_(1) == x_(1));
  P_ -= K*S*K.transpose();
  //P_ = 0.5 * (P_ + P_.transpose());

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}


//----------------------------------------------------------------------------------------------

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& m)
{
  const VectorXd& z = m.Measurement();
  const MatrixXd Zsig = STransformSigmaPointsIntoRadarMeasurementSpace(Xsig_pred_);

  // predict radar measurement and covariance
  const VectorXd z_pred = SPredictMean(Zsig, weights_);
  const MatrixXd S = SPredictCovariance(Zsig, z_pred, weights_) + R_radar_;

  const MatrixXd Tc = SCalcCrossCorrelationMatrix(Xsig_pred_, x_, Zsig, z_pred, weights_);
  const MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;
  SNormalizeAngle(z_diff, 1);

  x_ += K * z_diff;
  SNormalizeAngle(x_, 3);
  assert(x_(0) == x_(0));
  assert(x_(1) == x_(1));
  P_ -= K*S*K.transpose();
  // P_ = 0.5 * (P_ + P_.transpose());

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}


//==============================================================================================
