//==============================================================================================
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ukf.h"
#include "measurement_package.h"
// #include "ground_truth_package.h"
#include "tools.h"
//==============================================================================================
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector2d;
using std::vector;
//==============================================================================================




void check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " path/to/input.txt output.txt";

  // make sure the user has provided input and output files
  if (argc != 3)
  {
    cerr << usage_instructions << endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------------

template<typename T>
void check_file(T& file, const string& name)
{
  if (!file.is_open())
  {
    cerr << "Cannot open file: " << name << endl;
    exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------------

ofstream open_out_file(const std::string& file_name)
{
  ofstream out_file(file_name.c_str(), ofstream::out);
  check_file(out_file, file_name);
  return out_file;
}

//----------------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  check_arguments(argc, argv);

  const string in_file_name_ = argv[1];
  ifstream in_file_(in_file_name_.c_str(), ifstream::in);
  check_file(in_file_, in_file_name_);

  const string del = "\t";

  auto out_file = open_out_file(argv[2]);
  out_file << "pred_x" << del << "pred_y" << del << "pred_v" << del << "pred_yaw" << del << "pred_yaw_d";
  out_file << del << "meas_x" << del << "meas_y";
  out_file << del << "ground_truth_x" << del << "ground_truth_y";
  out_file << del << "ground_truth_vx" << del << "ground_truth_vy" << std::endl;

  auto out_file_nis_radar = open_out_file("nis_radar.txt");
  out_file_nis_radar << "NIS_Radar" << std::endl;
  auto out_file_nis_lidar = open_out_file("nis_lidar.txt");
  out_file_nis_lidar << "NIS_Lidar" << std::endl;

  /**********************************************
   *  Set Measurements                          *
   **********************************************/

  vector<MeasurementPackage> measurement_pack_list;
  vector<Eigen::VectorXd> estimation_list, ground_truth_list;
  string line;

  // prep the measurement packages (each line represents a measurement at a
  // timestamp)
  while (getline(in_file_, line))
  {
    string sensor_type;
    istringstream iss(line);
    long timestamp;

    // reads first element from the current line
    iss >> sensor_type;

    if (sensor_type == "L")
    {
      // laser measurement
      // read measurements at this timestamp
      VectorXd measurement = VectorXd(2);
      float lx;
      float ly;
      iss >> lx >> ly >> timestamp;
      measurement << lx, ly;
      measurement_pack_list.push_back(
        MeasurementPackage(timestamp, MeasurementPackage::LASER, measurement));
    }
    else if (sensor_type == "R")
    {
      // radar measurement
      // read measurements at this timestamp
      VectorXd measurement = VectorXd(3);
      float rho, theta, rho_dot;
      iss >> rho >> theta >> rho_dot >> timestamp;
      measurement << rho, theta, rho_dot;
      measurement_pack_list.push_back(
        MeasurementPackage(timestamp, MeasurementPackage::RADAR, measurement));
    }

    float px, py, vx, vy;
    iss >> px >> py >> vx >> vy;
    VectorXd g(4);
    g << px, py, vx, vy;
    ground_truth_list.push_back(g);
  }

  // Create a UKF instance
  UKF ukf;

  // start filtering from the second frame (the speed is unknown in the first
  // frame)
  assert(ground_truth_list.size() == measurement_pack_list.size());

  for (int i = 0; i < measurement_pack_list.size(); ++i)
  {
    const auto& m = measurement_pack_list[i];
    const auto& g = ground_truth_list[i];
    // Call the UKF-based fusion
    ukf.ProcessMeasurement(m);

    const double x = ukf.x_(0);
    const double y = ukf.x_(1);
    const double v = ukf.x_(2);
    const double yaw = ukf.x_(3);
    const double yaw_d = ukf.x_(4);

    // output the estimation
    out_file << x << del << y << del << v << del << yaw << del << yaw_d;

    const double vx = v*cos(yaw);
    const double vy = v*sin(yaw);
    VectorXd e(4);
    e << x, y, vx, vy;
    estimation_list.push_back(e);

    // output the measurements
    if (m.IsLaserMeasurement())
    {
      const double& px = m.Measurement()(0);
      const double& py = m.Measurement()(1);
      out_file << del << px << del << py;
      out_file_nis_lidar << ukf.NIS_laser_ << std::endl;
    }
    else if (m.IsRadarMeasurement())
    {
      // output the estimation in the cartesian coordinates
      const double& rho = m.Measurement()(0);
      const double& phi = m.Measurement()(1);
      const double px = rho * cos(phi);
      const double py = rho * sin(phi);
      out_file << del << px << del << py; // p1_meas
      out_file_nis_radar << ukf.NIS_radar_ << std::endl;
    }

    out_file << del << g(0) << del << g(1) << del << g(2) << del << g(3) << std::endl;
  }

  const auto rmse = Tools::CalculateRMSE(estimation_list, ground_truth_list);
  std::cout << "RMSE" << std::endl << rmse << std::endl;

  cout << "Done!" << endl;
  return 0;
}


//==============================================================================================
