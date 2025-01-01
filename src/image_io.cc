/**
 * @file image_io.cc
 * @author William Chan <root@williamchan.me>
 */
#include "image_io.h"
#include "worker.h"
#include "mat.h"
#include "utility.h"
#include "image_io.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace image {
    Napi::Value ImreadAsync(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        ARG_CHECK(info.Length() < 1, "Expected at least 1 argument");
        ARG_CHECK(!info[0].IsString(), "First argument must be a string (filename)");
        ARG_CHECK(info.Length() > 1 && !info[1].IsNumber(), "Second argument must be a number (flags)");

        std::string filename = info[0].As<Napi::String>().Utf8Value();
        int flags = cv::IMREAD_COLOR;
        if (info.Length() > 1 && info[1].IsNumber()) {
            flags = info[1].As<Napi::Number>().Int32Value();
        }
        auto execute = [filename, flags]() -> cv::Mat {
            if (!std::filesystem::exists(filename)) {
                throw std::runtime_error("File does not exist");
            }
            cv::Mat mat = cv::imread(filename, flags);
            if (mat.empty()) {
                throw std::runtime_error("Failed to read image");
            }
            return mat;
        };

        auto ok = [](Napi::Env env, cv::Mat& mat) -> Napi::Value {
            return Mat::NewInstance(env, mat);
        };

        auto* worker = new AsyncWorker<cv::Mat>(env, execute, ok);
        worker->Queue();
        return worker->GetPromise();
    }

    Napi::Value Imread(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        ARG_CHECK(info.Length() < 1, "Expected at least 1 argument");
        ARG_CHECK(!info[0].IsString(), "First argument must be a string (filename)");
        ARG_CHECK(info.Length() > 1 && !info[1].IsNumber(), "Second argument must be a number (flags)");

        std::string filename = info[0].As<Napi::String>().Utf8Value();
        int flags = cv::IMREAD_COLOR;
        if (info.Length() > 1 && info[1].IsNumber()) {
            flags = info[1].As<Napi::Number>().Int32Value();
        }

        cv::Mat mat = cv::imread(filename, flags);
        if (mat.empty()) {
            Napi::Error::New(env, "Failed to read image").ThrowAsJavaScriptException();
            return env.Null();
        }

        return Mat::NewInstance(env, mat);
    }

    Napi::Value ImdecodeAsync(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        ARG_CHECK(info.Length() < 1, "Expected at least 1 argument");
        ARG_CHECK(!info[0].IsBuffer(), "First argument must be a Buffer");
        ARG_CHECK(info.Length() > 1 && !info[1].IsNumber(), "Second argument must be a number (flags)");

        Napi::Buffer<uchar> buffer = info[0].As<Napi::Buffer<uchar>>();
        std::vector<uchar> buffer_data(buffer.Data(), buffer.Data() + buffer.Length());

        int flags = cv::IMREAD_COLOR;
        if (info.Length() > 1 && info[1].IsNumber()) {
            flags = info[1].As<Napi::Number>().Int32Value();
        }

        auto execute = [buffer_data, flags]() -> cv::Mat {
            cv::Mat mat = cv::imdecode(buffer_data, flags);
            if (mat.empty()) {
                throw std::runtime_error("Failed to decode image");
            }
            return mat;
        };

        auto ok = [](Napi::Env env, cv::Mat& mat) -> Napi::Value {
            return Mat::NewInstance(env, mat);
        };

        auto* worker = new AsyncWorker<cv::Mat>(env, execute, ok);
        worker->Queue();
        return worker->GetPromise();
    }

    Napi::Value Imdecode(const Napi::CallbackInfo& info) {
        Napi::Env env = info.Env();

        ARG_CHECK(info.Length() < 1, "Expected at least 1 argument");
        ARG_CHECK(!info[0].IsBuffer(), "First argument must be a Buffer");
        ARG_CHECK(info.Length() > 1 && !info[1].IsNumber(), "Second argument must be a number (flags)");

        Napi::Buffer<uchar> buffer = info[0].As<Napi::Buffer<uchar>>();
        std::vector<uchar> buffer_data(buffer.Data(), buffer.Data() + buffer.Length());

        int flags = cv::IMREAD_COLOR;
        if (info.Length() > 1 && info[1].IsNumber()) {
            flags = info[1].As<Napi::Number>().Int32Value();
        }

        cv::Mat mat = cv::imdecode(buffer_data, flags);
        if (mat.empty()) {
            Napi::Error::New(env, "Failed to decode image").ThrowAsJavaScriptException();
            return env.Null();
        }

        return Mat::NewInstance(env, mat);
    }

}