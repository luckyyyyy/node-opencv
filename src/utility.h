/**
 * @file common.h
 * @author William Chan <root@williamchan.me>
 */
#pragma once
#include <napi.h>

#define ARG_CHECK(condition, errorMessage) \
    if (condition) { \
        Napi::TypeError::New(env, errorMessage).ThrowAsJavaScriptException(); \
        return env.Null(); \
    }

namespace utility {
    Napi::Value GetBuildInformation(const Napi::CallbackInfo& info);
    Napi::Value GetTickCount(const Napi::CallbackInfo& info);
    Napi::Value GetTickFrequency(const Napi::CallbackInfo& info);
} // namespace utility