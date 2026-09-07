package dev.yolomaster.app.ui.bench

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import dev.yolomaster.app.YoloMasterApp
import dev.yolomaster.app.model.BundledModel
import dev.yolomaster.app.model.ComputeChoice
import dev.yolomaster.app.system.ThermalMonitor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID
import java.util.concurrent.Executors
import kotlin.math.max

/** UI-facing state of the Bench tab (`BenchView.swift` @State, lines 116-147). */
data class BenchUi(
    val models: List<BundledModel> = emptyList(),
    val mode: BenchMode = BenchMode.Sweep,
    /** The active mode's results; the other mode's set is stashed in the ViewModel. */
    val results: List<BenchResult> = emptyList(),
    val phase: BenchPhase = BenchPhase.Idle,
    val running: Boolean = false,
    val paused: Boolean = false,
    val showAdvanced: Boolean = false,
    // sustained target + settings
    val selectedModel: BundledModel? = null,
    val selectedCompute: ComputeChoice = ComputeChoice.GPU,
    val minutes: Int = 3,
    val iters: Int = 50,
    val warmup: Int = 10,
    // live readouts
    val liveMs: Double = 0.0,
    val runDuration: Int = 0,
    val sparkSamples: List<Double> = emptyList(),
    val sparkThermal: List<Int> = emptyList(),
    val sparkBaseline: Double? = null,
    val expanded: Set<String> = emptySet(),
    val error: String? = null,
) {
    val fastest: BenchResult? get() = results.minByOrNull { it.coldMedian }
    /** Catalog order, models that have at least one result (`modelsWithResults`). */
    val modelsWithResults: List<String> get() = models.map { it.id }.filter { id -> results.any { it.modelId == id } }
    val maxMinutes: Int get() = BenchStats.maxSustainedMinutes(selectedCompute)
    val canStart: Boolean get() = models.isNotEmpty() && (mode == BenchMode.Sweep || selectedModel != null)
}

/** One-shot events the screen turns into haptics (the ViewModel has no View). */
sealed class BenchEvent {
    data object Started : BenchEvent()
    data object Cell : BenchEvent()
    data object Finished : BenchEvent()
    data class Failed(val message: String) : BenchEvent()
}

/**
 * Activity-scoped state of the Bench tab: survives tab switches (the screen is disposed when
 * the tab leaves, but a paused run and its results come back with it). The bench loop runs on
 * ONE dedicated thread ("ym-bench") because `Detector.open` pins the caller to the big cores.
 */
class BenchViewModel(app: Application) : AndroidViewModel(app) {
    private val catalog = YoloMasterApp.from(app).catalog
    val history: BenchHistory = YoloMasterApp.from(app).history
    val thermal = ThermalMonitor(app)

    private val _ui = MutableStateFlow(BenchUi())
    val ui: StateFlow<BenchUi> = _ui
    private val _events = MutableSharedFlow<BenchEvent>(extraBufferCapacity = 16)
    val events: SharedFlow<BenchEvent> = _events

    private val engine = BenchEngine()
    private val benchDispatcher = Executors.newSingleThreadExecutor { r -> Thread(r, "ym-bench") }.asCoroutineDispatcher()
    private var control = BenchControl()
    private var job: Job? = null
    /** Bumped per start; a stale loop's late callbacks are ignored. */
    private var gen = 0
    private var stashed: List<BenchResult> = emptyList()   // the OTHER mode's results, kept hidden
    private var runThermalStart = 0
    private var runThermalPeak = 0

    init {
        thermal.start()
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) { catalog.discover() }
            _ui.update { it.copy(models = models, selectedModel = it.selectedModel ?: BundledModel.preferred(models)) }
        }
        // thermal peak tracked for the run's thermalPeak field (BenchView.swift:192-196)
        viewModelScope.launch { thermal.state.collect { s -> if (_ui.value.running) runThermalPeak = max(runThermalPeak, s.level) } }
    }

    // ---- settings ----------------------------------------------------------------------------

    /** Swap the result sets (`BenchView.swift:201-206`); the caller taps the light haptic. */
    fun setMode(m: BenchMode) {
        if (_ui.value.running || m == _ui.value.mode) return
        val tmp = _ui.value.results
        _ui.update { it.copy(mode = m, results = stashed, expanded = emptySet(), sparkSamples = emptyList(), sparkThermal = emptyList()) }
        stashed = tmp
    }

    fun toggleAdvanced() = _ui.update { it.copy(showAdvanced = !it.showAdvanced) }

    fun selectModel(m: BundledModel) = _ui.update {
        val c = if (m.cpuOnly) ComputeChoice.CPU else it.selectedCompute
        it.copy(selectedModel = m, selectedCompute = c, minutes = it.minutes.coerceAtMost(BenchStats.maxSustainedMinutes(c)))
    }

    /** Switching to CPU clamps the duration to the CPU cap (`BenchView.swift:301-304`). */
    fun selectCompute(c: ComputeChoice) = _ui.update {
        val unit = if (it.selectedModel?.cpuOnly == true) ComputeChoice.CPU else c
        it.copy(selectedCompute = unit, minutes = it.minutes.coerceAtMost(BenchStats.maxSustainedMinutes(unit)))
    }

    fun setMinutes(n: Int) = _ui.update { it.copy(minutes = n.coerceIn(1, it.maxMinutes)) }
    fun setIters(n: Int) = _ui.update { it.copy(iters = n.coerceIn(20, 200)) }
    fun setWarmup(n: Int) = _ui.update { it.copy(warmup = n.coerceIn(0, 50)) }
    fun toggleExpanded(modelId: String) = _ui.update { it.copy(expanded = if (modelId in it.expanded) it.expanded - modelId else it.expanded + modelId) }

    fun refreshModels() {
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) { catalog.discover(force = true) }
            _ui.update { u -> u.copy(models = models, selectedModel = u.selectedModel?.let { s -> models.firstOrNull { it.id == s.id } } ?: BundledModel.preferred(models)) }
        }
    }

    // ---- run control ---------------------------------------------------------------------------

    /** `start()` (`BenchView.swift:531-544`): clears the graph; a sweep also clears its results. */
    fun start() {
        val u = _ui.value
        if (u.running || !u.canStart) return
        control = BenchControl()
        val myGen = ++gen
        val level = thermal.state.value.level
        runThermalStart = level; runThermalPeak = level
        _ui.update {
            it.copy(
                running = true, paused = false, error = null, liveMs = 0.0, runDuration = 0,
                sparkSamples = emptyList(), sparkThermal = emptyList(), sparkBaseline = null,
                results = if (it.mode == BenchMode.Sweep) emptyList() else it.results,
                expanded = if (it.mode == BenchMode.Sweep) emptySet() else it.expanded,
            )
        }
        _events.tryEmit(BenchEvent.Started)
        val sink = Sink(myGen)
        val ctl = control
        job = viewModelScope.launch(benchDispatcher) {
            if (u.mode == BenchMode.Sweep) {
                val ok = engine.runSweep(u.models, u.warmup, u.iters, ctl, sink)
                if (ok && myGen == gen) finishRun(sustained = false, durationSec = 0)
            } else {
                val m = u.selectedModel ?: return@launch
                val c = u.selectedCompute
                when (val out = engine.runSustained(m, c, u.minutes, u.warmup, u.iters, ctl, sink)) {
                    is SustainedOutcome.Completed -> if (myGen == gen) {
                        val r = out.result
                        _ui.update {
                            it.copy(
                                // smooth transition from the rolling window to the full-run graph
                                sparkSamples = r.sparkline, sparkThermal = r.thermal, liveMs = r.sustainedMedian ?: it.liveMs,
                                runDuration = out.durationSec,
                                results = it.results.filterNot { x -> x.modelId == r.modelId && x.compute == r.compute } + r,
                            )
                        }
                        finishRun(sustained = true, durationSec = out.durationSec)
                    }
                    is SustainedOutcome.Failed -> if (myGen == gen) {
                        _ui.update { it.copy(running = false, paused = false, phase = BenchPhase.Idle, error = out.message) }
                        _events.tryEmit(BenchEvent.Failed(out.message))
                    }
                    SustainedOutcome.Cancelled -> {}
                }
            }
        }
    }

    /** `stop()` (`BenchView.swift:546-551`): cancelled runs are not saved. */
    fun stop() {
        control.cancelled = true
        job?.cancel(); job = null
        gen++
        _ui.update { it.copy(running = false, paused = false, phase = BenchPhase.Idle) }
    }

    /** The caller taps the light haptic (`BenchView.swift:553-557`). */
    fun togglePause() {
        if (!_ui.value.running) return
        val p = !_ui.value.paused
        control.paused = p
        _ui.update { it.copy(paused = p) }
    }

    /**
     * `autoPause()` (`BenchView.swift:210-212`): the tab was hidden or the app backgrounded; the
     * run pauses so the bench never competes with Live/Photo inference. Paused time is excluded.
     */
    fun onHidden() {
        if (_ui.value.running && !_ui.value.paused) { control.paused = true; _ui.update { it.copy(paused = true) } }
    }

    /** Symmetric hook; as on iOS a paused run resumes only when the user taps play. */
    fun onShown() {}

    /** `finishRun` (`BenchView.swift:568-586`): save to history, success haptic. */
    private fun finishRun(sustained: Boolean, durationSec: Int) {
        val u = _ui.value
        if (u.results.isNotEmpty()) {
            val name = BenchStats.runName(
                sustained, u.results.lastOrNull()?.shortID, u.selectedCompute.label, u.minutes, u.modelsWithResults.size,
            )
            val end = thermal.state.value.level
            history.add(
                BenchRun(
                    id = UUID.randomUUID().toString(), name = name, dateMs = System.currentTimeMillis(),
                    mode = if (sustained) BenchMode.Sustained.label else BenchMode.Sweep.label,
                    durationSec = if (sustained) durationSec else 0, results = u.results,
                    thermalStart = runThermalStart, thermalEnd = end, thermalPeak = max(runThermalPeak, end),
                ),
            )
        }
        _ui.update { it.copy(running = false, paused = false, phase = BenchPhase.Done) }
        _events.tryEmit(BenchEvent.Finished)
    }

    /** The active mode's results as `benchmark.csv` text. */
    fun resultsCSV(): String = BenchHistory.resultsCSV(_ui.value.results)

    private inner class Sink(private val myGen: Int) : BenchSink {
        private inline fun live(block: () -> Unit) { if (myGen == gen) block() }
        override fun phase(p: BenchPhase) = live { _ui.update { it.copy(phase = p) } }
        override fun cell(r: BenchResult) = live { _ui.update { it.copy(results = it.results + r) }; _events.tryEmit(BenchEvent.Cell) }
        override fun baseline(coldMedian: Double) = live {
            _ui.update { it.copy(sparkBaseline = coldMedian, sparkSamples = emptyList(), sparkThermal = emptyList(), runDuration = 0) }
        }
        override fun tick(spark: List<Double>, thermal: List<Int>, liveMs: Double, elapsedS: Int) = live {
            _ui.update { it.copy(sparkSamples = spark, sparkThermal = thermal, liveMs = liveMs, runDuration = elapsedS) }
        }
        override fun thermalLevel(): Int = thermal.state.value.level
    }

    override fun onCleared() {
        control.cancelled = true
        job?.cancel()
        benchDispatcher.close()
        thermal.stop()
    }
}
