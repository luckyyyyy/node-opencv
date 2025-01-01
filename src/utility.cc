/**
 * @file image_io.cc
 * @author William Chan <root@williamchan.me>
 */
#include "utility.h"
#include <napi.h>
#include <opencv2/opencv.hpp>

namespace utility {

  Napi::Value GetBuildInformation(const Napi::CallbackInfo& info) {
      Napi::Env env = info.Env();
      return Napi::String::New(env, cv::getBuildInformation());
  }
  Napi::Value GetTickCount(const Napi::CallbackInfo& info) {
      Napi::Env env = info.Env();
      return Napi::Number::New(env, cv::getTickCount());
  }
  Napi::Value GetTickFrequency(const Napi::CallbackInfo& info) {
      Napi::Env env = info.Env();
      return Napi::Number::New(env, cv::getTickFrequency());
  }
} // namespace utility