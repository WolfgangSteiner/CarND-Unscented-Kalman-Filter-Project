//==============================================================================================
#include <iostream>
#include "ukf.h"
#include "tools.h"
//==============================================================================================
using Eigen::VectorXd;
using Eigen::MatrixXd;
//==============================================================================================

//==============================================================================================

static void SNormalizeAngle(double& phi)
{
  phi = atan2(sin(phi), cos(phi));
}

//----------------------------------------------------------------------------------------------------------------------

static Eigen::VectorXd SPolarToCartesian(double rho, double phi)
{
  auto result = VectorXd(2);
  result << rho * cos(phi), rho * sin(phi);
  return result;
}


//----------------------------------------------------------------------------------------------

static bool SIsRadarMeasurement(const MeasurementPackage& m)
{
  return m.sensor_type_ == MeasurementPackage::RADAR;
}


//----------------------------------------------------------------------------------------------

static bool SIsLidarMeasurement(const MeasurementPackage& m)
{
  return m.sensor_type_ == MeasurementPackage::LASER;
}


//----------------------------------------------------------------------------------------------

static VectorXd SPrepareWeightVector()
{
  const int n_aug = 7;
  const int n_sigma = 2 * n_aug + 1;
  const int lambda = 3 - n_aug;
  VectorXd w = 0.5 * VectorXd::Ones(n_sigma);
  w(0) = lambda;
  w /= (lambda + n_aug);
  return w;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SCalcCovariance(
  const MatrixXd& XSig,
  const VectorXd& x,
  const VectorXd& w,
  int norm_idx)
{
  MatrixXd S = MatrixXd::Zero(XSig.rows(), XSig.rows());

  for (int i = 0; i < XSig.cols(); ++i)
  {
    VectorXd x_diff = XSig.col(i) - x;
    if (norm_idx >= 0)
    {
      SNormalizeAngle(x_diff(norm_idx));
    }
    S += w(i) * x_diff * x_diff.transpose();
  }

  return S;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SCalcCrossCorrelationMatrix(
  const MatrixXd& Xsig, const VectorXd& x,
  const MatrixXd& Zsig, const VectorXd& z,
  const VectorXd& w,
  int norm_idx=1)
{
  MatrixXd Tc = MatrixXd::Zero(x.rows(), z.rows());

  for (int i = 0; i < Xsig.cols(); ++i)
  {
    VectorXd x_diff = Xsig.col(i) - x;
    VectorXd z_diff = Zsig.col(i) - z;
    SNormalizeAngle(x_diff(3));
    if (norm_idx >= 0)
    {
      SNormalizeAngle(z_diff(norm_idx));
    }
    Tc += w(i) * x_diff * z_diff.transpose();
  }

  return Tc;
}

//----------------------------------------------------------------------------------------------

static MatrixXd SGenerateAugmentedSigmaPoints(
  const VectorXd& x,
  const MatrixXd& P,
  double std_a,
  double std_yawdd)
{
  const int n_aug = 7;
  const int n_sigma = 2 * n_aug + 1;
  const int n_x = 5;
  const double lambda = 3 - n_aug;

  MatrixXd P_aug = MatrixXd::Zero(n_aug, n_aug);
  P_aug.topLeftCorner(P.rows(), P.cols()) << P;
  P_aug.bottomRightCorner(2,2) << std_a * std_a, 0, 0, std_yawdd * std_yawdd;

  VectorXd x_aug(n_aug);
  x_aug << x, 0, 0;

  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sigma);
  const float f = sqrt(lambda + n_aug);
  const MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.col(0) = x_aug;

  for (int i = 0; i < n_aug; ++i)
  {
    Xsig_aug.col(i + 1)         = x_aug + f * A.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - f * A.col(i);
  }

  return Xsig_aug;
}

//----------------------------------------------------------------------------------------------

static VectorXd SCTRVModel(const VectorXd& x, double dt)
{
  const double v   = x(2);
  const double psi = x(3);
  const double psi_d = x(4);
  const double nu_a = x(5);
  const double nu_psi_dd = x(6);
  const double dt2 = dt * dt;

  VectorXd v1(5,1), v2(5,1);

  if (std::abs(psi_d) <= 0.001)
  {
    v1 << v * cos(psi) * dt, v * sin(psi) * dt, 0, 0, 0;
  }
  else
  {
    const double delta_psi = psi_d * dt;
    const double psi2 = psi + delta_psi;
    v1 << v / psi_d * ( sin(psi2) - sin(psi)),
          v / psi_d * (-cos(psi2) + cos(psi)),
          0,
          delta_psi,
          0;
  }

  v2 << 0.5 * dt2 * cos(psi) * nu_a,
        0.5 * dt2 * sin(psi) * nu_a,
        dt * nu_a,
        0.5 * dt2 * nu_psi_dd,
        dt * nu_psi_dd;

  return x.head(5) + v1 + v2;
}


//----------------------------------------------------------------------------------------------

static MatrixXd SPredictSigmaPoints(const MatrixXd& Xsig_aug, double aDeltaT)
{
  MatrixXd Xsig_pred = MatrixXd(5, 15);

  for (int i = 0; i < 15; ++i)
  {
    Xsig_pred.col(i) = SCTRVModel(Xsig_aug.col(i), aDeltaT);
  }

  return Xsig_pred;
}


//----------------------------------------------------------------------------------------------

static VectorXd SCalcMean(const MatrixXd& XSig, const VectorXd& w)
{
  return XSig * w;
  //
  // VectorXd x_pred = VectorXd::Zero(XSig.rows());
  // for (int i = 0; i < XSig.cols(); ++i)
  // {
  //   x_pred += w(i) * XSig.col(i);
  // }
  //
  // return x_pred;
}

//----------------------------------------------------------------------------------------------

static VectorXd STransformSigmaPointIntoRadarMeasurementSpace(const VectorXd& x)
{
  const double px = x(0);
  const double py = x(1);
  const double v = x(2);
  const double psi = x(3);

  const double vx = cos(psi) * v;
  const double vy = sin(psi) * v;

  const double rho = sqrt(px*px + py*py);
  const double phi = atan2(py,px);
  const double rho_dot = (px*vx + py*vy) / std::max(rho,1.0e-5);

  VectorXd z_pred(3);
  z_pred << rho, phi, rho_dot;
  return z_pred;
}


//----------------------------------------------------------------------------------------------

static MatrixXd STransformSigmaPointsIntoRadarMeasurementSpace(const MatrixXd& aXsig_pred)
{
  const int n_sigma = aXsig_pred.cols();
  MatrixXd Zsig = MatrixXd(3, n_sigma);

  for (int i = 0; i < n_sigma; i++)
  {
    Zsig.col(i) = STransformSigmaPointIntoRadarMeasurementSpace(aXsig_pred.col(i));
  }

  return Zsig;
}


//----------------------------------------------------------------------------------------------

static MatrixXd STransformSigmaPointsIntoLidarMeasurementSpace(const MatrixXd& aXsig_pred)
{
  const int n_sigma = aXsig_pred.cols();
  MatrixXd Zsig = MatrixXd(2, n_sigma);

  for (int i = 0; i < n_sigma; i++)
  {
    Zsig.col(i) << aXsig_pred.col(i).head(2);
  }

  return Zsig;
}


//==============================================================================================

UKF::UKF()
: is_initialized_(false)
, use_laser_(true)
, use_radar_(false)
, n_x_(5)
, n_aug_(7)
, n_sigma_(2*n_aug_ + 1)
, n_z_radar_(3)
, n_z_lidar_(2)
, lambda_(3 - n_aug_)
{
  x_ = VectorXd::Zero(5);
  P_ = MatrixXd::Identity(5,5);

  weights_ = SPrepareWeightVector();

  std_a_ = 3.0;
  std_yawdd_ = M_PI / 4;

  R_radar_ = MatrixXd(3,3);
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

}


//----------------------------------------------------------------------------------------------

void UKF::ProcessMeasurement(const MeasurementPackage& measurement_pack)
{
  if (!is_initialized_)
  {
    Initialize(measurement_pack);
    is_initialized_ = true;
    return;
  }

  double delta_t = (measurement_pack.timestamp_ - previous_timestamp_) / 1.0e6;
  previous_timestamp_ = measurement_pack.timestamp_;

  Predict(delta_t);

  if (SIsLidarMeasurement(measurement_pack) && use_laser_)
  {
    UpdateLidar(measurement_pack);
  }
  else if (SIsRadarMeasurement(measurement_pack) && use_radar_)
  {
    UpdateRadar(measurement_pack);
  }
}


//----------------------------------------------------------------------------------------------

void UKF::Initialize(const MeasurementPackage& measurement_pack)
{
  const auto& z = measurement_pack.raw_measurements_;

  if (SIsRadarMeasurement(measurement_pack))
  {
    x_ << SPolarToCartesian(z(0), z(1)), z(2), 0.0, 0.0;
  }
  else if (SIsLidarMeasurement(measurement_pack))
  {
    x_ << z, 0.0, 0.0, 0.0;
  }

  previous_timestamp_ = measurement_pack.timestamp_;
}


//----------------------------------------------------------------------------------------------

void UKF::Predict(double delta_t)
{
  const MatrixXd Xsig = SGenerateAugmentedSigmaPoints(x_, P_, std_a_, std_yawdd_);
  Xsig_pred_ = SPredictSigmaPoints(Xsig, delta_t);

  x_ = SCalcMean(Xsig_pred_, weights_);
  P_ = SCalcCovariance(Xsig_pred_, x_, weights_, 3);
}


//----------------------------------------------------------------------------------------------

void UKF::UpdateLidar(const MeasurementPackage& meas_package)
{
  MatrixXd H_lidar(2,5);
  H_lidar << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  MatrixXd R_lidar(2,2);
  R_lidar << 0.0225, 0,
              0, 0.0225;

  const VectorXd& z = meas_package.raw_measurements_;
  const VectorXd z_pred = H_lidar * x_;
  const VectorXd z_diff = z - z_pred;
  const MatrixXd PHt = P_ * H_lidar.transpose();
  const MatrixXd S = H_lidar * PHt + R_lidar;
  const MatrixXd K = PHt * S.inverse();
  x_ += K * z_diff;
  P_ -= K * H_lidar * P_;

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}


//----------------------------------------------------------------------------------------------

void UKF::UpdateRadar(const MeasurementPackage& meas_package)
{
  MatrixXd R_radar(3,3);
  R_radar << 0.09, 0, 0, 0, 0.0009, 0, 0, 0, 0.09;
  const VectorXd& z = meas_package.raw_measurements_;
  const MatrixXd Zsig_pred = STransformSigmaPointsIntoRadarMeasurementSpace(Xsig_pred_);
  const VectorXd z_pred = SCalcMean(Zsig_pred, weights_);
  const MatrixXd S = SCalcCovariance(Zsig_pred, z_pred, weights_,1) + R_radar;
  const MatrixXd Tc = SCalcCrossCorrelationMatrix(Xsig_pred_, x_, Zsig_pred, z_pred, weights_);
  const MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z - z_pred;
  SNormalizeAngle(z_diff(1));

  x_ += K * z_diff;
//  SNormalizeAngle(x_(3));
  P_ -= K*S*K.transpose();

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}


//==============================================================================================
