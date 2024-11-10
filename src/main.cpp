/**
 * @file main.cpp
 * @author William Chan <root@williamchan.me>
 */
#include <napi.h>
#include "matrix.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
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


    return Matrix::Init(env, exports);
}

NODE_API_MODULE(opencv_addon, Init)
