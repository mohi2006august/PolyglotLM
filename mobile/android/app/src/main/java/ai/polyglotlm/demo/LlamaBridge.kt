package ai.polyglotlm.demo

import java.io.ByteArrayOutputStream

/** Receives generated text as UTF-8 byte chunks that always end on a code point boundary. */
interface TokenCallback {
    fun onBytes(bytes: ByteArray)

    fun isCancelled(): Boolean
}

/**
 * Thin Kotlin face of the llama.cpp JNI bridge in polyglot_jni.cpp. Not thread-safe.
 * All natives are instance methods so their JNI symbols are plain `Java_..._LlamaBridge_*`.
 */
class LlamaBridge private constructor() : AutoCloseable {
    private var handle: Long = 0L

    data class Result(val text: String, val tokens: Int, val seconds: Double) {
        val tokensPerSecond: Double get() = if (seconds > 0) tokens / seconds else 0.0
        val charsPerSecond: Double get() = if (seconds > 0) text.length / seconds else 0.0
    }

    fun generate(
        prompt: String,
        maxTokens: Int = 256,
        cancelled: () -> Boolean = { false },
        onText: (String) -> Unit,
    ): Result {
        check(handle != 0L) { "model already closed" }
        val all = ByteArrayOutputStream()
        val start = System.nanoTime()
        val tokens = nativeGenerate(handle, prompt, maxTokens, object : TokenCallback {
            override fun onBytes(bytes: ByteArray) {
                all.write(bytes)
                onText(String(bytes, Charsets.UTF_8))
            }

            override fun isCancelled(): Boolean = cancelled()
        })
        val seconds = (System.nanoTime() - start) / 1e9
        check(tokens >= 0) { "generation failed" }
        return Result(all.toString(Charsets.UTF_8.name()), tokens, seconds)
    }

    override fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }

    private external fun nativeGenerate(
        handle: Long,
        prompt: String,
        maxTokens: Int,
        callback: TokenCallback,
    ): Int

    private external fun nativeFree(handle: Long)

    private external fun nativeLoad(path: String, threads: Int, nCtx: Int): Long

    companion object {
        init {
            System.loadLibrary("polyglot")
        }

        fun load(path: String, threads: Int, nCtx: Int = 2048): LlamaBridge {
            val bridge = LlamaBridge()
            bridge.handle = bridge.nativeLoad(path, threads, nCtx)
            require(bridge.handle != 0L) { "could not load model at $path" }
            return bridge
        }
    }
}
