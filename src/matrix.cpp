/**
 * @file matrix.cpp
 * @author William Chan <root@williamchan.me>
 */

#include "matrix.h"
#include "async.h"

Napi::FunctionReference Matrix::constructor;

Napi::Object Matrix::Init(Napi::Env env, Napi::Object exports) {
    Napi::HandleScope scope(env);

    Napi::Function func = DefineClass(env, "Matrix", {
        InstanceMethod("matchTemplateAsync", &Matrix::MatchTemplateAsync),
        InstanceMethod("minMaxLocAsync", &Matrix::MinMaxLocAsync),
        StaticMethod("imdecodeAsync", &Matrix::ImdecodeAsync)
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("Matrix", func);
    return exports;
}

Matrix::Matrix(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Matrix>(info) {
}

Matrix::~Matrix() {
    mat.release();
}

Napi::Value Matrix::ImdecodeAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        throw Napi::Error::New(env, "Buffer argument required");
    }

    Napi::Buffer<uchar> buffer = info[0].As<Napi::Buffer<uchar>>();
    std::vector<uchar> buffer_data(buffer.Data(), buffer.Data() + buffer.Length());

    int flags = cv::IMREAD_COLOR;

    if (info.Length() > 1 && info[1].IsNumber()) {
        flags = info[1].As<Napi::Number>().Int32Value();
    }

    auto executeCallback = [buffer_data, flags]() -> cv::Mat {
        cv::Mat result = cv::imdecode(buffer_data, flags);
        if (result.empty()) {
            throw std::runtime_error("Failed to decode image");
        }
        return result;
    };

    auto resolveCallback = [](Napi::Env env, cv::Mat& result) -> Napi::Value {
        auto matrix = Matrix::constructor.New({});
        Matrix* unwrapped = Matrix::Unwrap(matrix);
        unwrapped->mat = result.clone();
        return matrix;
    };

    auto* worker = new OpenCVAsyncWorker<cv::Mat>(env, executeCallback, resolveCallback);
    worker->Queue();
    return worker->Promise();
}

Napi::Value Matrix::MatchTemplateAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        throw Napi::Error::New(env, "Template and method arguments required");
    }

    Matrix* templ = Napi::ObjectWrap<Matrix>::Unwrap(info[0].As<Napi::Object>());
    int method = info[1].As<Napi::Number>().Int32Value();

    cv::Mat img_clone = this->mat.clone();
    cv::Mat templ_clone = templ->mat.clone();

    auto executeCallback = [img_clone, templ_clone, method]() -> cv::Mat {
        cv::Mat result;
        cv::matchTemplate(img_clone, templ_clone, result, method);
        return result;
    };

    auto resolveCallback = [](Napi::Env env, cv::Mat& result) -> Napi::Value {
        auto matrix = Matrix::constructor.New({});
        Matrix* unwrapped = Matrix::Unwrap(matrix);
        unwrapped->mat = result.clone();
        return matrix;
    };

    auto* worker = new OpenCVAsyncWorker<cv::Mat>(env, executeCallback, resolveCallback);
    worker->Queue();
    return worker->Promise();
}

struct MinMaxResult {
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
};

Napi::Value Matrix::MinMaxLocAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    cv::Mat mat_clone = this->mat.clone();

    auto executeCallback = [mat_clone]() -> MinMaxResult {
        MinMaxResult result;
        cv::minMaxLoc(mat_clone, &result.minVal, &result.maxVal,
                      &result.minLoc, &result.maxLoc);
        return result;
    };

    auto resolveCallback = [](Napi::Env env, MinMaxResult& result) -> Napi::Value {
        auto obj = Napi::Object::New(env);
        obj.Set("minVal", Napi::Number::New(env, result.minVal));
        obj.Set("maxVal", Napi::Number::New(env, result.maxVal));

        auto minLocObj = Napi::Object::New(env);
        minLocObj.Set("x", Napi::Number::New(env, result.minLoc.x));
        minLocObj.Set("y", Napi::Number::New(env, result.minLoc.y));
        obj.Set("minLoc", minLocObj);

        auto maxLocObj = Napi::Object::New(env);
        maxLocObj.Set("x", Napi::Number::New(env, result.maxLoc.x));
        maxLocObj.Set("y", Napi::Number::New(env, result.maxLoc.y));
        obj.Set("maxLoc", maxLocObj);

        return obj;
    };

    auto* worker = new OpenCVAsyncWorker<MinMaxResult>(env, executeCallback, resolveCallback);
    worker->Queue();
    return worker->Promise();
}
