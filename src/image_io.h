/**
 * @file image_io.h
 * @author William Chan <root@williamchan.me>
 */
#pragma once
#include <napi.h>

namespace image {
    Napi::Value ImreadAsync(const Napi::CallbackInfo& info);
    Napi::Value Imread(const Napi::CallbackInfo& info);
    Napi::Value ImdecodeAsync(const Napi::CallbackInfo& info);
    Napi::Value Imdecode(const Napi::CallbackInfo& info);
}