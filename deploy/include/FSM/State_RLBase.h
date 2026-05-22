// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <vector>
#include <chrono>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        start_time = std::chrono::steady_clock::now();

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        if (has_upper_body_)
        {
            int rl_joint_count = env->robot->data.joint_stiffness.size();
            upper_body_q0_.clear();
            for (size_t i = 0; i < upper_body_kp_.size(); ++i)
            {
                int motor_idx = rl_joint_count + i;
                auto & motor = lowcmd->msg_.motor_cmd()[motor_idx];
                motor.kp() = upper_body_kp_[i];
                motor.kd() = upper_body_kd_[i];
                motor.dq() = motor.tau() = 0;
                upper_body_q0_.push_back(motor.q());
            }
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

protected:
    bool has_upper_body_ = false;
    std::vector<float> upper_body_kp_;
    std::vector<float> upper_body_kd_;
    std::vector<float> upper_body_qs_;
    std::vector<float> upper_body_q0_;

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
