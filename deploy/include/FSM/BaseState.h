// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <utility>
#include <chrono>

inline boost::bimap<int, std::string> FSMStringMap;

class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state) 
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {
    }

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {}

    double duration() const {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        return elapsed.count();
    }

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
private:
    int state_;
};

using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmMap     = std::unordered_map<std::string, FsmFactory>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = __factory_##Derived;                                \
        }                                                                               \
    } __registrar_instance_##Derived;                                                   \
    
