#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

class MeasurementPackage
{
public:
  enum SensorType{
    LASER,
    RADAR
  };

public:
  MeasurementPackage(long aTimeStamp, SensorType aSensorType, const Eigen::VectorXd& aMeasurement)
  : timestamp_(aTimeStamp)
  , sensor_type_(aSensorType)
  , raw_measurements_(aMeasurement)
  {}

  bool IsRadarMeasurement() const { return sensor_type_ == RADAR; }
  bool IsLaserMeasurement() const { return sensor_type_ == LASER; }
  long TimeStamp() const { return timestamp_; }
  const Eigen::VectorXd& Measurement() const { return raw_measurements_; }

public:
  long timestamp_;
  SensorType sensor_type_;
  Eigen::VectorXd raw_measurements_;
};

#endif /* MEASUREMENT_PACKAGE_H_ */
