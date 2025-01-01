/**
 * @file main.cc
 * @author William Chan <root@williamchan.me>
 */
#include "image_io.h"
#include "mat.h"
#include "utility.h"

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    Mat::Exports(env, exports);
    exports.Set("imreadAsync", Napi::Function::New(env, image::ImreadAsync));
    exports.Set("imread", Napi::Function::New(env, image::Imread));
    exports.Set("imdecodeAsync", Napi::Function::New(env, image::ImdecodeAsync));
    exports.Set("imdecode", Napi::Function::New(env, image::Imdecode));

    exports.Set("getBuildInformation", Napi::Function::New(env, utility::GetBuildInformation));
    exports.Set("getTickCount", Napi::Function::New(env, utility::GetTickCount));
    exports.Set("getTickFrequency", Napi::Function::New(env, utility::GetTickFrequency));

    exports.Set(Napi::String::New(env, "CV_VERSION"), Napi::String::New(env, CV_VERSION));
    exports.Set(Napi::String::New(env, "TM_SQDIFF"), Napi::Number::New(env, cv::TM_SQDIFF));
    exports.Set(Napi::String::New(env, "TM_SQDIFF_NORMED"), Napi::Number::New(env, cv::TM_SQDIFF_NORMED));
    exports.Set(Napi::String::New(env, "TM_CCORR"), Napi::Number::New(env, cv::TM_CCORR));
    exports.Set(Napi::String::New(env, "TM_CCORR_NORMED"), Napi::Number::New(env, cv::TM_CCORR_NORMED));
    exports.Set(Napi::String::New(env, "TM_CCOEFF"), Napi::Number::New(env, cv::TM_CCOEFF));
    exports.Set(Napi::String::New(env, "TM_CCOEFF_NORMED"), Napi::Number::New(env, cv::TM_CCOEFF_NORMED));

    exports.Set(Napi::String::New(env, "IMREAD_UNCHANGED"), Napi::Number::New(env, cv::IMREAD_UNCHANGED));
    exports.Set(Napi::String::New(env, "IMREAD_GRAYSCALE"), Napi::Number::New(env, cv::IMREAD_GRAYSCALE));
    exports.Set(Napi::String::New(env, "IMREAD_COLOR"), Napi::Number::New(env, cv::IMREAD_COLOR));
    exports.Set(Napi::String::New(env, "IMREAD_ANYDEPTH"), Napi::Number::New(env, cv::IMREAD_ANYDEPTH));
    exports.Set(Napi::String::New(env, "IMREAD_ANYCOLOR"), Napi::Number::New(env, cv::IMREAD_ANYCOLOR));
    exports.Set(Napi::String::New(env, "IMREAD_LOAD_GDAL"), Napi::Number::New(env, cv::IMREAD_LOAD_GDAL));
    exports.Set(Napi::String::New(env, "IMREAD_REDUCED_GRAYSCALE_2"), Napi::Number::New(env, cv::IMREAD_REDUCED_GRAYSCALE_2));
    exports.Set(Napi::String::New(env, "IMREAD_REDUCED_COLOR_2"), Napi::Number::New(env, cv::IMREAD_REDUCED_COLOR_2));
    exports.Set(Napi::String::New(env, "IMREAD_REDUCED_GRAYSCALE_4"), Napi::Number::New(env, cv::IMREAD_REDUCED_GRAYSCALE_4));
    exports.Set(Napi::String::New(env, "IMREAD_REDUCED_COLOR_4"), Napi::Number::New(env, cv::IMREAD_REDUCED_COLOR_4));
    exports.Set(Napi::String::New(env, "IMREAD_REDUCED_GRAYSCALE_8"), Napi::Number::New(env, cv::IMREAD_REDUCED_GRAYSCALE_8));
    exports.Set(Napi::String::New(env, "IMREAD_REDUCED_COLOR_8"), Napi::Number::New(env, cv::IMREAD_REDUCED_COLOR_8));
    exports.Set(Napi::String::New(env, "IMREAD_IGNORE_ORIENTATION"), Napi::Number::New(env, cv::IMREAD_IGNORE_ORIENTATION));

    return exports;
}

NODE_API_MODULE(addon, Init)