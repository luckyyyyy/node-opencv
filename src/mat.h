/**
 * @file mat.h
 * @author William Chan <root@williamchan.me>
 */
#pragma once
#include <napi.h>
#include <opencv2/opencv.hpp>


class Mat : public Napi::ObjectWrap<Mat> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    static Napi::Object NewInstance(Napi::Env env, cv::Mat mat);
    Napi::Value MatchTemplateAsync(const Napi::CallbackInfo& info);
    Napi::Value MinMaxLocAsync(const Napi::CallbackInfo& info);
    Napi::Value Release(const Napi::CallbackInfo& info);
    Napi::Value GetCols(const Napi::CallbackInfo& info);
    Napi::Value GetRows(const Napi::CallbackInfo& info);
    Napi::Value GetData(const Napi::CallbackInfo& info);
    Napi::Value GetSize(const Napi::CallbackInfo& info);
    cv::Mat mat;

    Mat(const Napi::CallbackInfo& info);
    cv::Mat GetMat() {
        return mat;
    }

private:
    static Napi::FunctionReference constructor;
    // Instance methods
};
