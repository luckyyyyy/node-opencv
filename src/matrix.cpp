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
        InstanceMethod("matchTemplate", &Matrix::MatchTemplate),
        InstanceMethod("minMaxLocAsync", &Matrix::MinMaxLocAsync),
        InstanceMethod("release", &Matrix::Release),
        StaticMethod("imdecodeAsync", &Matrix::ImdecodeAsync),
        StaticMethod("imdecode", &Matrix::Imdecode),
        StaticMethod("imread", &Matrix::Imread),
        StaticMethod("imreadAsync", &Matrix::ImreadAsync),

        InstanceAccessor<&Matrix::GetCols>("cols"),
        InstanceAccessor<&Matrix::GetRows>("rows"),
        InstanceAccessor<&Matrix::GetData>("data")
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
    if (!mat.empty()) {
        mat.release();
    }
}

Napi::Value Matrix::Imdecode(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    try {
        if (info.Length() < 1) {
            throw std::runtime_error("Buffer argument required");
        }
        if (!info[0].IsBuffer()) {
            throw std::runtime_error("Buffer argument must be a Buffer");
        }
        Napi::Buffer<uchar> buffer = info[0].As<Napi::Buffer<uchar>>();
        std::vector<uchar> buffer_data(buffer.Data(), buffer.Data() + buffer.Length());

        cv::Mat result = cv::imdecode(buffer_data, cv::IMREAD_COLOR);

        if (result.empty()) {
            throw std::runtime_error("Failed to decode image");
        }
        auto matrix = Matrix::constructor.New({});
        Matrix* unwrapped = Matrix::Unwrap(matrix);
        unwrapped->mat = std::move(result);
        return matrix;
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}



Napi::Value Matrix::ImdecodeAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        throw std::runtime_error("Buffer argument required");
    }

    Napi::Buffer<uchar> buffer = info[0].As<Napi::Buffer<uchar>>();
    std::vector<uchar> buffer_data(buffer.Data(), buffer.Data() + buffer.Length());

    int flags = cv::IMREAD_COLOR;
    if (info.Length() > 1 && info[1].IsNumber()) {
        flags = info[1].As<Napi::Number>().Int32Value();
    }

    return AsyncWorker<cv::Mat>::Execute(
        env,
        [buffer_data = std::move(buffer_data), flags]() {
            cv::Mat result = cv::imdecode(buffer_data, flags);
            if (result.empty()) {
                throw std::runtime_error("Failed to decode image");
            }
            return result;
        },
        [](Napi::Env env, const cv::Mat& result) {
            auto matrix = Matrix::constructor.New({});
            Matrix* unwrapped = Matrix::Unwrap(matrix);
            unwrapped->mat = std::move(result);
            return matrix;
        }
    );
}

Napi::Value Matrix::Imread(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    try {
        if (info.Length() < 1) {
            throw std::runtime_error("Filename argument required");
        }
        if (!info[0].IsString()) {
            throw std::runtime_error("Filename argument must be a string");
        }
        std::string filename = info[0].As<Napi::String>().Utf8Value();
        int flags = cv::IMREAD_COLOR;

        if (info.Length() > 1 && info[1].IsNumber()) {
            flags = info[1].As<Napi::Number>().Int32Value();
        }

        cv::Mat result = cv::imread(filename, flags);
        if (result.empty()) {
            throw std::runtime_error("Failed to load image Path: " + filename);
        }
        auto matrix = Matrix::constructor.New({});
        Matrix* unwrapped = Matrix::Unwrap(matrix);
        unwrapped->mat = std::move(result);
        return matrix;
        return env.Null();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value Matrix::ImreadAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1) {
        throw std::runtime_error("Filename argument required");
    }

    std::string filename = info[0].As<Napi::String>().Utf8Value();
    int flags = cv::IMREAD_COLOR;

    if (info.Length() > 1 && info[1].IsNumber()) {
        flags = info[1].As<Napi::Number>().Int32Value();
    }

    return AsyncWorker<cv::Mat>::Execute(
        env,
        [filename, flags]() {
            cv::Mat result = cv::imread(filename, flags);
            if (result.empty()) {
                throw std::runtime_error("Failed to load image: " + filename);
            }
            return result;
        },
        [](Napi::Env env, const cv::Mat& result) {
            auto matrix = Matrix::constructor.New({});
            Matrix* unwrapped = Matrix::Unwrap(matrix);
            unwrapped->mat = result.clone();
            return matrix;
        }
    );
}

Napi::Value Matrix::MatchTemplateAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

    if (info.Length() < 2) {
        deferred.Reject(Napi::TypeError::New(env, "Expected 2 arguments").Value());
        return deferred.Promise();
    }

    if (!info[0].As<Napi::Object>().InstanceOf(Matrix::constructor.Value())) {
        deferred.Reject(Napi::TypeError::New(env, "First argument must be a Matrix instance").Value());
        return deferred.Promise();
    }

    if (!info[1].IsNumber()) {
        deferred.Reject(Napi::TypeError::New(env, "Second argument must be a number").Value());
        return deferred.Promise();
    }

    Matrix* templ = Napi::ObjectWrap<Matrix>::Unwrap(info[0].As<Napi::Object>());
    int method = info[1].As<Napi::Number>().Int32Value();

    auto sourceMat = this->mat;
    auto templateMat = templ->mat;

    return AsyncWorker<cv::Mat>::Execute(
        env,
        [sourceMat, templateMat, method]() {
            cv::Mat result;
            cv::matchTemplate(sourceMat, templateMat, result, method);
            return result;
        },
        [](Napi::Env env, const cv::Mat& result) {
            auto matrix = Matrix::constructor.New({});
            Matrix* unwrapped = Matrix::Unwrap(matrix);
            unwrapped->mat = result.clone();
            return matrix;
        }
    );
}

Napi::Value Matrix::MatchTemplate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2) {
        Napi::TypeError::New(env, "Expected 2 arguments").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[0].As<Napi::Object>().InstanceOf(Matrix::constructor.Value())) {
        Napi::TypeError::New(env, "First argument must be a Matrix instance").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (!info[1].IsNumber()) {
        Napi::TypeError::New(env, "Second argument must be a number").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Matrix* templ = Napi::ObjectWrap<Matrix>::Unwrap(info[0].As<Napi::Object>());
    int method = info[1].As<Napi::Number>().Int32Value();

    cv::Mat result;
    cv::matchTemplate(mat, templ->mat, result, method);

    auto matrix = Matrix::constructor.New({});
    Matrix* unwrapped = Matrix::Unwrap(matrix);
    unwrapped->mat = std::move(result);
    return matrix;
}

Napi::Value Matrix::Release(const Napi::CallbackInfo& info) {
    mat.release();
    return info.This();
}

struct MinMaxResult {
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
};

Napi::Value Matrix::MinMaxLocAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto self = this;

    return AsyncWorker<MinMaxResult>::Execute(
        env,
        [self]() {
            MinMaxResult result;
            cv::minMaxLoc(self->mat, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc);
            return result;
        },
        [](Napi::Env env, const MinMaxResult& result) {
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
        }
    );
}

Napi::Value Matrix::GetCols(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), mat.cols);
}

Napi::Value Matrix::GetRows(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), mat.rows);
}

Napi::Value Matrix::GetData(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (mat.empty()) {
        return Napi::Buffer<uchar>::New(env, 0);
    }

    size_t dataSize = mat.total() * mat.elemSize();
    if (dataSize == 0) {
        return Napi::Buffer<uchar>::New(env, 0);
    }

    return Napi::Buffer<uchar>::New(
        env,
        mat.data,
        dataSize
    );
}

