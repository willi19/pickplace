//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <iostream>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {
      
    /// create world
    world_ = std::make_unique<raisim::World>();
    is_init_ = true;


    raisim::Mat<3, 3> inertia;
    inertia.setIdentity();
    const raisim::Vec<3> com = {0, 0, 0};

    object_ = static_cast<raisim::Mesh*>(world_->addMesh(resourceDir_+"/objects/bottle/bottle.obj", 
                                          1.0, 
                                          inertia, 
                                          com, 
                                          1.0, 
                                          "", 
                                          raisim::COLLISION(2), 
                                          raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
    object_->setName("object");
    object_->setBodyType(raisim::BodyType::DYNAMIC);
    

    /// add objects
    xarm6_ = world_->addArticulatedSystem(resourceDir_+"/assembly/xarm6/xarm6_allegro_wrist_mounted_rotate.urdf", "",{},raisim::COLLISION(0),raisim::COLLISION(2)|raisim::COLLISION(63));
    xarm6_->setName("xarm6");
    xarm6_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
  
    world_->addGround(-0.0525);

    /// set gravity
    // const raisim::Vec<3> gravity = {0.0, 0.0, 0.0};
    // world_->setGravity(gravity);
    // print("Updated gravity:", world.get_gravity())  # 출력: [0.0, 0.0, 0.0]

    /// get robot data
    gcDim_ = xarm6_->getGeneralizedCoordinateDim();
    gvDim_ = xarm6_->getDOF();
    nJoints_ = gvDim_;
    
    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of xarm6
    
    // gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8;//, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    gc_init_ <<     0.412873,
                  -0.214403,
                  0.0391116,
                      0.9409,
                  -0.0144877,
                -0.000299348,
                -0.33142079,
                0.54690431,
                0.13329243,
                0.0847668, 
                -0.18806428,
                0.59743658,
                0.1704287,
                0.10751154,
                -0.11053761,
                0.37680732,
                0.13505902,
                0.08466482,
                0.92694671,
                0.49415258,
                0.31955628,
                0.18234833;
    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(1000.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(100);
    xarm6_->setPdGains(jointPgain, jointDgain);
    xarm6_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34;
    actionDim_ = nJoints_+7; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(xarm6_);
      xarm6_vis = server_->addVisualArticulatedSystem("xarm6_vis", resourceDir_+"/assembly/xarm6/xarm6_allegro_wrist_mounted_rotate.urdf");
      object_vis = server_->addVisualMesh("object_vis", resourceDir_+"/objects/bottle/bottle.obj");
    }
  }

  void init() final { }

  void reset() final {
    // xarm6_->setState(gc_init_, gv_init_);
    // updateObservation();
    is_init_ = true;
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget_ = action.cast<double>().segment(7, nJoints_);
    // std::cout<<pTarget_[19]<<std::endl;
    // for(int i=22-3; i<22; i++) pTarget_[i] *= 10;
    // std::cout<<pTarget_<<std::endl;
    // pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    // pTarget12_ += actionMean_;
    // pTarget_.tail(nJoints_) = pTarget12_;

    if (is_init_) {
      xarm6_->setGeneralizedCoordinate(pTarget_);
      object_->setPosition(action[0], action[1], action[2]);
      object_->setOrientation(action[3], action[4], action[5], action[6]);
      is_init_ = false;
    }

    else
      xarm6_->setPdTarget(pTarget_, vTarget_);

    xarm6_vis->setGeneralizedCoordinate(pTarget_);
    object_vis->setPosition(action[0],action[1],action[2]);
    object_vis->setOrientation(action[3], action[4], action[5], action[6]);
    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }

    //updateObservation();

    rewards_.record("torque", xarm6_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    return rewards_.sum();
  }

  void updateObservation() {
    xarm6_->getState(gc_, gv_);
    std::cout<<gc_<<std::endl;
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[0]; 
    quat[1] = gc_[1]; 
    quat[2] = gc_[2]; 
    quat[3] = gc_[3];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        gc_.tail(12), /// joint angles
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
        gv_.tail(12); /// joint velocity
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);
    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  int gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* xarm6_;
  raisim::ArticulatedSystemVisual* xarm6_vis;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_;
  raisim::Mesh *object_;
  raisim::Visuals *object_vis;
  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
  bool is_init_ = false;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}

