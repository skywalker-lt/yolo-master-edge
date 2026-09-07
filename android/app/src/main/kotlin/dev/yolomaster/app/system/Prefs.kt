package dev.yolomaster.app.system

import android.content.Context
import android.content.SharedPreferences
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext

/** `@AppStorage` equivalents. Single SharedPreferences file, read synchronously (tiny). */
object Prefs {
    private const val FILE = "yolomaster"
    const val ALLOW_CPU = "allowCPU"

    fun get(ctx: Context): SharedPreferences = ctx.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    /** iOS default is false (Core ML CPU crashes); on Android CPU is the certified path -> true. */
    fun allowCPU(ctx: Context): Boolean = get(ctx).getBoolean(ALLOW_CPU, true)
    fun setAllowCPU(ctx: Context, v: Boolean) = get(ctx).edit().putBoolean(ALLOW_CPU, v).apply()
}

/** A Compose state mirrored into SharedPreferences, updated when the key changes elsewhere. */
@Composable
fun rememberBoolPref(key: String, default: Boolean): MutableState<Boolean> {
    val ctx = LocalContext.current
    val prefs = remember { Prefs.get(ctx) }
    val state = remember { mutableStateOf(prefs.getBoolean(key, default)) }
    androidx.compose.runtime.DisposableEffect(key) {
        val l = SharedPreferences.OnSharedPreferenceChangeListener { p, k -> if (k == key) state.value = p.getBoolean(key, default) }
        prefs.registerOnSharedPreferenceChangeListener(l)
        onDispose { prefs.unregisterOnSharedPreferenceChangeListener(l) }
    }
    return object : MutableState<Boolean> by state {
        override var value: Boolean
            get() = state.value
            set(v) { state.value = v; prefs.edit().putBoolean(key, v).apply() }
    }
}
