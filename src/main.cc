/**
 * @file main.cc
 * @author William Chan <root@williamchan.me>
 */
#include "image_io.h"
#include "mat.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Mat::Init(env, exports);
    exports.Set("imreadAsync", Napi::Function::New(env, image::ImreadAsync));
    exports.Set("imread", Napi::Function::New(env, image::Imread));
    exports.Set("imdecodeAsync", Napi::Function::New(env, image::ImdecodeAsync));
    exports.Set("imdecode", Napi::Function::New(env, image::Imdecode));
    exports.Set(Napi::String::New(env, "TM_SQDIFF"), Napi::Number::New(env, cv::TM_SQDIFF));
    exports.Set(Napi::String::New(env, "TM_SQDIFF_NORMED"), Napi::Number::New(env, cv::TM_SQDIFF_NORMED));
    exports.Set(Napi::String::New(env, "TM_CCORR"), Napi::Number::New(env, cv::TM_CCORR));
    exports.Set(Napi::String::New(env, "TM_CCORR_NORMED"), Napi::Number::New(env, cv::TM_CCORR_NORMED));
    exports.Set(Napi::String::New(env, "TM_CCOEFF"), Napi::Number::New(env, cv::TM_CCOEFF));
    exports.Set(Napi::String::New(env, "TM_CCOEFF_NORMED"), Napi::Number::New(env, cv::TM_CCOEFF_NORMED));

    exports.Set(Napi::String::New(env, "IMREAD_COLOR"), Napi::Number::New(env, cv::IMREAD_COLOR));
    exports.Set(Napi::String::New(env, "IMREAD_GRAYSCALE"), Napi::Number::New(env, cv::IMREAD_GRAYSCALE));
    exports.Set(Napi::String::New(env, "IMREAD_UNCHANGED"), Napi::Number::New(env, cv::IMREAD_UNCHANGED));
    exports.Set(Napi::String::New(env, "IMREAD_ANYDEPTH"), Napi::Number::New(env, cv::IMREAD_ANYDEPTH));
    exports.Set(Napi::String::New(env, "IMREAD_ANYCOLOR"), Napi::Number::New(env, cv::IMREAD_ANYCOLOR));

    return exports;
}

NODE_API_MODULE(addon, Init)