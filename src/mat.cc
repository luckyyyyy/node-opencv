/**
 * @file mat.h
 * @author William Chan <root@williamchan.me>
 */
#include "mat.h"
#include "worker.h"
#include "utility.h"

struct MatData {
    Napi::FunctionReference constructor;
};

Mat::Mat(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Mat>(info) {}

Napi::Object Mat::NewInstance(Napi::Env env, cv::Mat mat) {
    Napi::EscapableHandleScope scope(env);

    MatData* data = env.GetInstanceData<MatData>();

    Napi::Object obj = data->constructor.New({});
    Mat* wrapper = Napi::ObjectWrap<Mat>::Unwrap(obj);
    wrapper->mat = mat;

    return scope.Escape(obj).ToObject();
}

Napi::Value Mat::MatchTemplateAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    ARG_CHECK(info.Length() < 2, "Expected at least 2 arguments");
    ARG_CHECK(!info[0].IsObject(), "First argument must be a Mat object (template)");
    ARG_CHECK(!info[1].IsNumber(), "Second argument must be a number (method)");

    Mat* templ = Napi::ObjectWrap<Mat>::Unwrap(info[0].As<Napi::Object>());
    int method = info[1].As<Napi::Number>().Int32Value();

    cv::Mat srcMat = this->mat;
    cv::Mat templMat = templ->GetMat();

    auto execute = [srcMat, templMat, method]() -> cv::Mat {
        cv::Mat result;
        cv::matchTemplate(srcMat, templMat, result, method);
        if (result.empty()) {
            throw std::runtime_error("Match template failed");
        }
        return result;
    };

    auto ok = [](Napi::Env env, cv::Mat& result) -> Napi::Value {
        return Mat::NewInstance(env, result);
    };

    auto* worker = new AsyncWorker<cv::Mat>(env, execute, ok);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value Mat::MinMaxLocAsync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    cv::Mat srcMat = this->mat;

    auto execute = [srcMat]() -> std::tuple<double, double, cv::Point, cv::Point> {
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(srcMat, &minVal, &maxVal, &minLoc, &maxLoc);
        return std::make_tuple(minVal, maxVal, minLoc, maxLoc);
    };

    auto ok = [](Napi::Env env, std::tuple<double, double, cv::Point, cv::Point>& result) -> Napi::Value {
        Napi::Object obj = Napi::Object::New(env);

        obj.Set("minVal", std::get<0>(result));
        obj.Set("maxVal", std::get<1>(result));

        Napi::Object minLoc = Napi::Object::New(env);
        minLoc.Set("x", std::get<2>(result).x);
        minLoc.Set("y", std::get<2>(result).y);
        obj.Set("minLoc", minLoc);

        Napi::Object maxLoc = Napi::Object::New(env);
        maxLoc.Set("x", std::get<3>(result).x);
        maxLoc.Set("y", std::get<3>(result).y);
        obj.Set("maxLoc", maxLoc);

        return obj;
    };

    auto* worker = new AsyncWorker<std::tuple<double, double, cv::Point, cv::Point>>(env, execute, ok);
    worker->Queue();
    return worker->GetPromise();
}


Napi::Value Mat::Release(const Napi::CallbackInfo& info) {
    mat.release();
    return info.Env().Undefined();
}

Napi::Value Mat::GetCols(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), mat.cols);
}

Napi::Value Mat::GetRows(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), mat.rows);
}

Napi::Value Mat::GetSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto size = mat.size();
    auto obj = Napi::Object::New(env);
    obj.Set("width", Napi::Number::New(env, size.width));
    obj.Set("height", Napi::Number::New(env, size.height));
    return obj;
}


Napi::Value Mat::GetData(const Napi::CallbackInfo& info) {
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

Napi::Object Mat::Exports(Napi::Env env, Napi::Object exports) {

    Napi::Function func = DefineClass(env, "Mat", {
        InstanceMethod("matchTemplateAsync", &Mat::MatchTemplateAsync),
        InstanceMethod("minMaxLocAsync", &Mat::MinMaxLocAsync),
        InstanceMethod("release", &Mat::Release),
        InstanceAccessor<&Mat::GetCols>("cols"),
        InstanceAccessor<&Mat::GetRows>("rows"),
        InstanceAccessor<&Mat::GetData>("data"),
        InstanceAccessor<&Mat::GetSize>("size")

    });

    MatData* data = new MatData();
    data->constructor = Napi::Persistent(func);
    data->constructor.SuppressDestruct();

    env.SetInstanceData<MatData>(data);

    exports.Set("Mat", func);
    return exports;
}