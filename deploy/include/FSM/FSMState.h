#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }


        // --- 新增：键盘控制逻辑 ---
        // 假设我们定义：键盘 '1' 站立，'2' 走路，'3 4 5' mimic动作，'0' 退出
        std::map<std::string, std::string> kb_shortcuts = {
            {"0", "Passive"},
            {"1", "FixStand"},
            {"2", "Velocity"},
            {"3", "Mimic_TaiChi"},
            {"4", "Mimic_PickUpBox"},
            {"5", "Mimic_Lafan1_Dance1_Subject2"},
            {"up", "FixStand"},    // 额外支持物理方向键
            {"down", "Passive"}
        };
        
        for(auto const& [key_str, target_name] : kb_shortcuts)
        {
            if(FSMStringMap.right.count(target_name))
            {
                int target_id = FSMStringMap.right.at(target_name);
                
                registered_checks.emplace_back(
                    std::make_pair(
                        [key_str]()->bool { 
                            if (!keyboard) return false;

                            // 逻辑：
                            // 1. keyboard->on_pressed 必须为 true（表示这是按下瞬间）
                            // 2. keyboard->key() 的内容必须匹配我们定义的键（如 "1" 或 "up"）
                            return keyboard->on_pressed && (keyboard->key() == key_str); 
                        },
                        target_id
                    )
                );
                spdlog::info("State_{}: Registered KB Shortcut [{}] -> {}", state_string, key_str, target_name);
            }
        }
        
        //-------------------------

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};