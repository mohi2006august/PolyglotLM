// JNI bridge between the Android app and llama.cpp (FR-07).
//
// Written against the llama.h API that uses llama_model_load_from_file / llama_init_from_model /
// llama_vocab_* and the sampler chain. Not compiled in CI; see mobile/android/README.md.

#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "polyglot"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct Session {
    llama_model *model = nullptr;
    int threads = 4;
    int n_ctx = 2048;
};

// Same plain template the SFT stage uses when the tokenizer has no chat template.
const char *kPlainTemplate = "### Instruction:\n%s\n\n### Response:\n";

std::string format_prompt(llama_model *model, const std::string &user) {
    const char *tmpl = llama_model_chat_template(model, nullptr);
    if (tmpl != nullptr) {
        llama_chat_message message{"user", user.c_str()};
        std::vector<char> buf(user.size() * 2 + 512);
        int32_t n = llama_chat_apply_template(tmpl, &message, 1, true, buf.data(), buf.size());
        if (n > static_cast<int32_t>(buf.size())) {
            buf.resize(n);
            n = llama_chat_apply_template(tmpl, &message, 1, true, buf.data(), buf.size());
        }
        if (n > 0) return std::string(buf.data(), n);
    }
    std::vector<char> buf(user.size() + 64);
    int n = snprintf(buf.data(), buf.size(), kPlainTemplate, user.c_str());
    return std::string(buf.data(), n);
}

// Length of the longest prefix of `s` that ends on a complete UTF-8 code point. Indic and Urdu
// characters are multi-byte, and a token piece can end halfway through one.
size_t complete_utf8_prefix(const std::string &s) {
    for (size_t back = 1; back <= 4 && back <= s.size(); ++back) {
        auto c = static_cast<unsigned char>(s[s.size() - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte
        size_t need = (c & 0x80) == 0      ? 1
                      : (c & 0xE0) == 0xC0 ? 2
                      : (c & 0xF0) == 0xE0 ? 3
                      : (c & 0xF8) == 0xF0 ? 4
                                           : 1;
        return back >= need ? s.size() : s.size() - back;
    }
    return s.size();
}

void emit(JNIEnv *env, jobject callback, jmethodID on_bytes, const std::string &chunk) {
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(chunk.size()));
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(chunk.size()),
                            reinterpret_cast<const jbyte *>(chunk.data()));
    env->CallVoidMethod(callback, on_bytes, bytes);
    env->DeleteLocalRef(bytes);
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_ai_polyglotlm_demo_LlamaBridge_nativeLoad(JNIEnv *env, jobject, jstring path, jint threads,
                                                jint n_ctx) {
    llama_backend_init();
    const char *c_path = env->GetStringUTFChars(path, nullptr);
    llama_model_params params = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(c_path, params);
    env->ReleaseStringUTFChars(path, c_path);
    if (model == nullptr) {
        LOGE("failed to load model");
        return 0;
    }
    auto *session = new Session{model, threads, n_ctx};
    LOGI("model loaded, %d threads, ctx %d", threads, n_ctx);
    return reinterpret_cast<jlong>(session);
}

// Streams UTF-8 chunks to callback.onBytes(byte[]); returns the number of generated tokens,
// or -1 on error. A fresh context per call keeps turns independent.
extern "C" JNIEXPORT jint JNICALL
Java_ai_polyglotlm_demo_LlamaBridge_nativeGenerate(JNIEnv *env, jobject, jlong handle,
                                                    jstring prompt, jint max_tokens,
                                                    jobject callback) {
    auto *session = reinterpret_cast<Session *>(handle);
    if (session == nullptr) return -1;
    jclass cb_class = env->GetObjectClass(callback);
    jmethodID on_bytes = env->GetMethodID(cb_class, "onBytes", "([B)V");
    jmethodID is_cancelled = env->GetMethodID(cb_class, "isCancelled", "()Z");

    const char *c_prompt = env->GetStringUTFChars(prompt, nullptr);
    std::string text = format_prompt(session->model, c_prompt);
    env->ReleaseStringUTFChars(prompt, c_prompt);

    const llama_vocab *vocab = llama_model_get_vocab(session->model);
    int n_prompt = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_prompt, true, true) < 0) {
        LOGE("tokenization failed");
        return -1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = session->n_ctx;
    cparams.n_batch = std::max<uint32_t>(n_prompt, 512);
    cparams.n_threads = session->threads;
    cparams.n_threads_batch = session->threads;
    llama_context *ctx = llama_init_from_model(session->model, cparams);
    if (ctx == nullptr) return -1;

    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    std::string pending;
    int generated = 0;
    llama_token token;
    while (generated < max_tokens && n_prompt + generated < session->n_ctx) {
        if (llama_decode(ctx, batch) != 0) {
            LOGE("decode failed at token %d", generated);
            break;
        }
        token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        char piece[256];
        int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n > 0) pending.append(piece, n);
        size_t ready = complete_utf8_prefix(pending);
        if (ready > 0) {
            emit(env, callback, on_bytes, pending.substr(0, ready));
            pending.erase(0, ready);
        }
        ++generated;
        if (env->CallBooleanMethod(callback, is_cancelled)) break;
        batch = llama_batch_get_one(&token, 1);
    }
    if (!pending.empty()) emit(env, callback, on_bytes, pending);

    llama_sampler_free(sampler);
    llama_free(ctx);
    return generated;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_polyglotlm_demo_LlamaBridge_nativeFree(JNIEnv *, jobject, jlong handle) {
    auto *session = reinterpret_cast<Session *>(handle);
    if (session == nullptr) return;
    llama_model_free(session->model);
    delete session;
    llama_backend_free();
}
