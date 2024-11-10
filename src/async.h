/**
 * @file async.h
 * @author William Chan <root@williamchan.me>
 */
#pragma once
#include <napi.h>
#include <opencv2/opencv.hpp>
#include <functional>

template<typename T>
class OpenCVAsyncWorker : public Napi::AsyncWorker {
public:
    using ExecuteCallback = std::function<T()>;
    using ResolveCallback = std::function<Napi::Value(Napi::Env, T&)>;

    OpenCVAsyncWorker(Napi::Env env,
                     ExecuteCallback executeCallback,
                     ResolveCallback resolveCallback)
        : AsyncWorker(env)
        , deferred(Napi::Promise::Deferred::New(env))
        , executeCallback(executeCallback)
        , resolveCallback(resolveCallback) {}

    void Execute() override {
        try {
            result = executeCallback();
        } catch (const std::exception& e) {
            SetError(e.what());
        }
    }

    void OnOK() override {
        Napi::HandleScope scope(Env());
        auto value = resolveCallback(Env(), result);
        deferred.Resolve(value);
    }

    Napi::Promise Promise() {
        return deferred.Promise();
    }

private:
    Napi::Promise::Deferred deferred;
    ExecuteCallback executeCallback;
    ResolveCallback resolveCallback;
    T result;
};
