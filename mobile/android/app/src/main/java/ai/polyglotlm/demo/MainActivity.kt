package ai.polyglotlm.demo

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Offline chat: one prompt box, streamed answer, measured speed. No network permission. */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor()
    private val cancelled = AtomicBoolean(false)
    private var bridge: LlamaBridge? = null
    private var running = false

    private lateinit var status: TextView
    private lateinit var output: TextView
    private lateinit var prompt: EditText
    private lateinit var generate: Button
    private lateinit var scroll: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        status = findViewById(R.id.status)
        output = findViewById(R.id.output)
        prompt = findViewById(R.id.prompt)
        generate = findViewById(R.id.generate)
        scroll = findViewById(R.id.scroll)
        generate.setOnClickListener { if (running) cancelled.set(true) else startGeneration() }
        loadModel()
    }

    private fun modelFile(): File? =
        getExternalFilesDir(null)?.listFiles { file -> file.extension == "gguf" }?.firstOrNull()

    private fun loadModel() {
        val file = modelFile()
        if (file == null) {
            val dir = getExternalFilesDir(null)?.absolutePath ?: "<app files dir>"
            status.text = getString(R.string.missing_model, "$dir/")
            return
        }
        worker.execute {
            // Big cores only: mid-range SoCs have 2-4 of them; more threads slow decoding down.
            val threads = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)
            val loaded = runCatching { LlamaBridge.load(file.absolutePath, threads) }
            runOnUiThread {
                loaded.onSuccess {
                    bridge = it
                    status.text = getString(R.string.ready) + " · " + file.name
                    generate.isEnabled = true
                }.onFailure { status.text = it.message }
            }
        }
    }

    private fun startGeneration() {
        val model = bridge ?: return
        val text = prompt.text.toString().trim()
        if (text.isEmpty()) return
        running = true
        cancelled.set(false)
        output.text = ""
        generate.text = getString(R.string.stop)
        worker.execute {
            val result = runCatching {
                model.generate(text, maxTokens = 256, cancelled = { cancelled.get() }) { chunk ->
                    runOnUiThread {
                        output.append(chunk)
                        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
                    }
                }
            }
            runOnUiThread {
                result.onSuccess {
                    status.text = getString(
                        R.string.speed, it.tokensPerSecond, it.charsPerSecond, it.tokens,
                    )
                }.onFailure { status.text = it.message }
                running = false
                generate.text = getString(R.string.generate)
            }
        }
    }

    override fun onDestroy() {
        cancelled.set(true)
        worker.execute { bridge?.close() }
        worker.shutdown()
        super.onDestroy()
    }
}
