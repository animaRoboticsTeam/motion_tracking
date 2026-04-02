#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    // auto action = env->action_manager->processed_actions();
    // for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
    //     lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    // }


    // 1. 获取 RL 策略输出 (包含全身 29 个关节)
    auto action = env->action_manager->processed_actions();
    
    // 2. 计算插值系数
    float elapsed_time = this->duration();
     
    float alpha = std::min(elapsed_time / transition_time, 1.0f);

    // 3. 遍历动作映射表
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) 
    {
        int motor_id = env->robot->data.joint_ids_map[i];
        float target_q = action[i]; // RL 策略希望达到的位置

        // --- 核心修改：分部位逻辑 ---
        
        // 如果是腿部或腰部 (0-14)，直接下发，不进行插值
        if (motor_id < 15) 
        {
            lowcmd->msg_.motor_cmd()[motor_id].q() = target_q;
        }
        // 如果是手臂 (15-28)，执行平滑插值
        else 
        {
            float start_q = initial_q[motor_id];
            // 从跳舞结束的位置缓慢移动到 RL 期望的位置
            float blended_q = start_q * (1.0f - alpha) + target_q * alpha;
            lowcmd->msg_.motor_cmd()[motor_id].q() = blended_q;
        }
    }
}