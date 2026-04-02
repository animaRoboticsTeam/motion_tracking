// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        start_time = std::chrono::steady_clock::now();

        // 记录切换瞬间的初始关节位置
        // 假设机器人有 29 个电机
        auto& ms = FSMState::lowstate->msg_.motor_state(); 
        for (int i = 0; i < ms.size(); ++i)
        {
            // 直接从 motor_state 结构体中读取当前的弧度值 q()
            initial_q[i] = ms[i].q();
        }
        
        interpolation_done = false;
        transition_time = 1.5f;
        // ...


        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
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

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // --- 修改开始：插值相关变量 ---
    float initial_q[29];           // 存储进入状态时的关节位置
    bool interpolation_done = false;
    float transition_time = 1.5f; // 过渡时间设为 1.5 秒
    // --- 修改结束 ---
};

REGISTER_FSM(State_RLBase)
