/**
 * @file worker.h
 * @author William Chan <root@williamchan.me>
 */
#pragma once
#include <napi.h>
#include <functional>
#include <utility>

template <typename R>
class AsyncWorker : public Napi::AsyncWorker {
public:
    using ExecuteFn = std::function<R()>;
    using Ok = std::function<Napi::Value(Napi::Env, R&)>;

    AsyncWorker(Napi::Env env, ExecuteFn execute, Ok convFunc)
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
          execute(std::move(execute)), convFunc(std::move(convFunc)) {}

    void Execute() override {
        try {
            result = execute();
        } catch (const std::exception& e) {
            SetError(e.what());
        }
    }

    void OnOK() override {
        Napi::Value value = convFunc(Env(), result);
        deferred.Resolve(value);
    }

    void OnError(const Napi::Error& e) override {
        deferred.Reject(e.Value());
    }

    Napi::Promise GetPromise() {
        return deferred.Promise();
    }

private:
    Napi::Promise::Deferred deferred;
    ExecuteFn execute;
    Ok convFunc;
    R result;
};