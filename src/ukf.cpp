//======================================================================================================================
#include <iostream>
#include "ukf.h"
#include "tools.h"
//======================================================================================================================

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
: use_laser_(true)
, use_radar_(true)
{
  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}


//----------------------------------------------------------------------------------------------------------------------

UKF::~UKF() {}

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

  if (m.sensor_type_ == MeasurementPackage::LASER && use_laser_)
  {

  }
  else if (use_radar_)
  {

  }
}


//----------------------------------------------------------------------------------------------------------------------

static VectorXd SCTRVModel(const VectorXd& aX_aug, float aDeltaT)
{
  const double px = aX_aug(0);
  const double py = aX_aug(1);
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


//----------------------------------------------------------------------------------------------------------------------

static MatrixXd SGenerateSigmaPoints(const VectorXd& aX, const MatrixXd& aP)
{
  //set state dimension
  int n_x = 5;

  //define spreading parameter
  double lambda = 3 - n_x;

  //create sigma point matrix
  MatrixXd Xsig = MatrixXd(n_x, 2 * n_x + 1);

  //calculate square root of P
  MatrixXd A = aP.llt().matrixL();

  const float f = sqrt(lambda + n_x);
  Xsig.col(0) = aX;

  for (int i = 0; i < n_x; ++i)
  {
    Xsig.col(i + 1) = aX + f * A.col(i);
    Xsig.col(i + 1 + n_x) = aX - f * A.col(i);
  }

  return Xsig;
}

//----------------------------------------------------------------------------------------------------------------------

MatrixXd UKF::SGenerateAugmentedSigmaPoints(MatrixXd* Xsig_out)
{
  const int n_x = 5;
  const int n_aug = 7;
  const int n_sigma = 2 * n_aug + 1;

  //Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = 0.2;

  //Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = 0.2;

  //define spreading parameter
  const double lambda = 3 - n_aug;

  //set example state
  VectorXd x = VectorXd(n_x);
  x << 5.7441,
    1.3800,
    2.2049,
    0.5015,
    0.3528;

  //create example covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
    -0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
    0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
    -0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
    -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(7, 7);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);

  P_aug.topLeftCorner(5, 5) = P;
  P_aug(5, 5)  = std_a;
  P_aug(6, 6)  = std_yawdd;

  x_aug.head(5) = x;
  x_aug.tail(2) << 0, 0;

  const float f = sqrt(lambda + n_aug);
  const MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < 7; ++i) {
    Xsig_aug.col(i + 1) = x_aug + f * A.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - f * A.col(i);
  }
}

//----------------------------------------------------------------------------------------------------------------------

void UKF::Initialize(const MeasurementPackage& m)
{
  if (m.sensor_type_ == MeasurementPackage::RADAR)
  {
    const double rho = m.raw_measurements_(0);
    const double phi = m.raw_measurements_(1, 0);
    const double rho_dot = m.raw_measurements_(2, 0);
    auto pos = Tools::PolarToCartesian(rho, phi);3
    auto vel = Tools::PolarToCartesian(rho_dot, phi);
    x_ << pos(0, 0), pos(1, 0), vel(0, 0), vel(1, 0);
  }
  else if (m.sensor_type_ == MeasurementPackage::LASER)
  {
    x_ << m.raw_measurements_(0, 0), m.raw_measurements_(1, 0), 0.0, 0.0;
  }
}

//----------------------------------------------------------------------------------------------------------------------

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */

void UKF::Prediction(double delta_t)
{
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
}


//----------------------------------------------------------------------------------------------------------------------

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */

void UKF::UpdateLidar(const MeasurementPackage& m)
{
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}


//----------------------------------------------------------------------------------------------------------------------

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(const MeasurementPackage& m)
{
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}


//----------------------------------------------------------------------------------------------------------------------

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out)
{
  //set state dimension
  int n_x = 5;

  //set augmented dimension
  int n_aug = 7;

  //create example sigma point matrix

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  double delta_t = 0.1; //time diff in sec
  VectorXd xk = Xsig_aug.col(0).head(5);

  for (int i = 0; i < 2 * n_aug + 1; ++i)
  {
    Xsig_pred.col(i) = CTRVModel(Xsig_aug.col(i), delta_t);
  }
}



//======================================================================================================================
