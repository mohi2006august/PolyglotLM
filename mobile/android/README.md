# PolyglotLM Android demo (FR-07)

An offline chat app that runs the quantised PolyglotLM model on the phone with llama.cpp. It has
no `INTERNET` permission: the model is pushed with `adb` and every token is generated on-device.

> **Status: unverified.** This project has not been compiled or run in CI. The JNI bridge is
> written against the llama.h API that uses `llama_model_load_from_file`,
> `llama_init_from_model`, `llama_vocab_*` and the sampler chain; if your llama.cpp checkout is
> older or newer and a symbol changed, the compiler will point at it in `polyglot_jni.cpp`.

## Build

Requirements: Android Studio (or the command-line SDK), NDK r27, CMake 3.22.1, JDK 17.

```bash
cd mobile/android
git clone https://github.com/ggml-org/llama.cpp app/src/main/cpp/llama.cpp
# pin it for reproducible builds:
git -C app/src/main/cpp/llama.cpp checkout <tag>
```

Open `mobile/android` in Android Studio and run the `app` configuration, or with a Gradle
installation:

```bash
gradle wrapper && ./gradlew :app:assembleRelease
```

`app/src/main/cpp/llama.cpp` is git-ignored; record the tag you built against in your release
notes, next to the GGUF hash.

## Get a model onto the phone

```bash
poly quant --recipe recipes/indic-3b.yaml --q Q4_K_M   # -> artifacts/indic-3b/gguf/*.gguf
adb install app/build/outputs/apk/release/app-release.apk
adb push artifacts/indic-3b/gguf/indic-3b-sft-Q4_K_M.gguf \
  /sdcard/Android/data/ai.polyglotlm.demo/files/
```

The app loads the first `.gguf` it finds in its files directory.

## Measure speed on the device

The status line shows tokens/s and characters/s after every answer. For the PRD target
(>= 8 tok/s for a 3B Q4 model on a mid-range phone) use `llama-bench` built for Android and feed
its JSON to the tokenizer-aware report, which converts tok/s into words/s per language:

```bash
adb shell /data/local/tmp/llama-bench -m /data/local/tmp/model.gguf -p 128 -n 64 -t 4 -o json > bench.json
poly bench --recipe recipes/indic-3b.yaml --from-json bench.json
```

## Design notes

- Token pieces are emitted only at UTF-8 code point boundaries: Indic and Urdu characters are
  multi-byte and a single token can end mid-character.
- The prompt is formatted with the model's own chat template when the GGUF carries one, else
  with the plain `### Instruction / ### Response` template the SFT stage uses, so the app sees the
  same format the model was tuned on.
- Generation uses 2-4 threads (the big cores of a mid-range SoC); more threads usually slow
  decoding down on phones.
- Streaming output and status updates are announced as polite live regions for TalkBack.
