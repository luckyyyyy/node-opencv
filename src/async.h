/**
 * @file async.h
 * @author William Chan <root@williamchan.me>
 */
#pragma once
#include <napi.h>
#include <thread>
#include <functional>
#include <future>
#include "thread_pool.h"

template<typename Result>
class AsyncWorker {
public:
    using ExecuteCallback = std::function<Result()>;
    using ResolveCallback = std::function<Napi::Value(Napi::Env env, const Result&)>;

    static Napi::Promise Execute(
        Napi::Env env,
        ExecuteCallback executeCallback,
        ResolveCallback resolveCallback
    ) {
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);

        auto tsfn = Napi::ThreadSafeFunction::New(
            env,
            Napi::Function::New(env, [](const Napi::CallbackInfo& info){}),
            "AsyncWorker",
            0,
            1
        );

        static ThreadPool& pool = ThreadPool::getInstance();

        pool.enqueue([
            executeCallback = std::move(executeCallback),
            resolveCallback = std::move(resolveCallback),
            tsfn,
            deferred
        ]() {
            try {
                Result result = executeCallback();

                auto callback = [
                    result = std::move(result),
                    resolveCallback = std::move(resolveCallback),
                    deferred
                ](Napi::Env env, Napi::Function) {
                    try {
                        Napi::Value jsResult = resolveCallback(env, result);
                        deferred.Resolve(jsResult);
                    } catch (const std::exception& e) {
                        deferred.Reject(Napi::Error::New(env, e.what()).Value());
                    }
                };

                tsfn.BlockingCall(callback);
            } catch (const std::exception& e) {
                auto callback = [e = std::string(e.what()), deferred](Napi::Env env, Napi::Function) {
                    deferred.Reject(Napi::Error::New(env, e).Value());
                };
                tsfn.BlockingCall(callback);
            }

            tsfn.Release();
        });

        return deferred.Promise();
    }
};
