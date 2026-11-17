// The app's own text in three languages - English, Simplified Chinese, Traditional Chinese -
// and the language switch in Settings (1.2.1).
//
// How it works
//   - Every text the app shows is written in the code in ENGLISH and passed through `tr`:
//         Text(tr("No photos"))            tr("Saved {0} images to Photos", n)
//     The English text is the key. `tr` returns it as it is while the app is in English and
//     looks it up in `Translations` (Translations.swift) otherwise; a text with no entry
//     there comes out in English, so a missing translation is never a blank.
//   - `{0}`, `{1}` ... stand for the values handed to `tr`. A translation may use them in
//     another order.
//   - Settings > Language (`AppLanguage`) is kept in UserDefaults. "System" follows the
//     languages of the device, in the user's order.
//   - A view that shows text observes the `Localizer`:
//         @ObservedObject private var localizer = Localizer.shared
//     so its text is drawn again the moment the language changes. A view without that line
//     keeps its old text until something else redraws it.
//
// What is NOT translated, on purpose: CPU, GPU, Neural Engine, Core ML, NMS, IoU, conf,
// kpt, FPS, ms and the other terms of the trade; model names; class names (they are the
// model's own output and are drawn into saved images); licence names and copyright lines;
// the licence notice in Settings (its English text is the notice); CSV exports.
//
// The tables are Swift code and not .strings files because the language is switched while
// the app runs: iOS loads a .strings table for the language the APP was started in and
// offers no supported way to load another one into SwiftUI's own text lookup. The two
// permission prompts and nothing else are .strings (en.lproj, zh-Hans.lproj, zh-Hant.lproj
// / InfoPlist.strings): iOS shows those itself, in the device's language.
import Foundation
import SwiftUI

/// The choice in Settings > Language.
enum AppLanguage: String, CaseIterable, Identifiable {
    case system
    case english = "en"
    case simplifiedChinese = "zh-Hans"
    case traditionalChinese = "zh-Hant"

    var id: String { rawValue }

    /// What the picker shows: a language by its own name, whatever language the app is in,
    /// so that someone who cannot read the current one still finds theirs.
    var displayName: String {
        switch self {
        case .system: return tr("System")
        case .english: return "English"
        case .simplifiedChinese: return "简体中文"
        case .traditionalChinese: return "繁體中文"
        }
    }
}

/// The language the app's text is in.
enum TextLanguage {
    case english, simplifiedChinese, traditionalChinese
}

/// Holds the language choice and tells the views when it changes.
final class Localizer: ObservableObject {
    static let shared = Localizer()

    private static let defaultsKey = "appLanguage"

    /// The choice in Settings. Changed on the main thread only (`select`).
    @Published private(set) var choice: AppLanguage = .system

    // `tr` is also called off the main thread (a model import, a benchmark, a photo run
    // build their messages there), so the language it reads is behind a lock.
    private let lock = NSLock()
    private var current: TextLanguage = .english

    private init() {
        let stored = UserDefaults.standard.string(forKey: Localizer.defaultsKey) ?? ""
        let restored = AppLanguage(rawValue: stored) ?? .system
        choice = restored
        current = Localizer.textLanguage(for: restored)
    }

    /// The language to show text in right now. Safe from any thread.
    var language: TextLanguage {
        lock.lock()
        defer { lock.unlock() }
        return current
    }

    /// Settings > Language. Main thread.
    func select(_ new: AppLanguage) {
        guard new != choice else { return }
        UserDefaults.standard.set(new.rawValue, forKey: Localizer.defaultsKey)
        let text = Localizer.textLanguage(for: new)
        lock.lock()
        current = text
        lock.unlock()
        choice = new    // published: the observing views draw their text again
    }

    /// The text language a choice stands for. "System" takes the first of the device's
    /// languages, in the user's order, that the app has: the rule iOS itself applies to an
    /// app, and it includes the per-app language set in the Settings app, which iOS hands
    /// over as the first entry of this list. A device with neither English nor Chinese
    /// among its languages gets English.
    static func textLanguage(for choice: AppLanguage) -> TextLanguage {
        switch choice {
        case .english: return .english
        case .simplifiedChinese: return .simplifiedChinese
        case .traditionalChinese: return .traditionalChinese
        case .system:
            for identifier in Locale.preferredLanguages {
                if let match = textLanguage(forIdentifier: identifier) { return match }
            }
            return .english
        }
    }

    /// "zh-Hans-CN", "zh-Hant-HK", "zh-TW", "yue-Hant-HK", "en-GB" ... -> the app's
    /// language for it, nil for a language the app does not have. Chinese written in
    /// Traditional characters is told by its script, or by its region where an identifier
    /// names no script (Taiwan, Hong Kong, Macau); every other Chinese is Simplified.
    static func textLanguage(forIdentifier identifier: String) -> TextLanguage? {
        let parts = identifier.lowercased()
            .split(whereSeparator: { $0 == "-" || $0 == "_" })
            .map(String.init)
        guard let code = parts.first else { return nil }
        if code == "en" { return .english }
        guard code == "zh" || code == "yue" else { return nil }
        if parts.contains("hans") { return .simplifiedChinese }
        if parts.contains("hant") { return .traditionalChinese }
        if parts.contains("tw") || parts.contains("hk") || parts.contains("mo") { return .traditionalChinese }
        return code == "yue" ? .traditionalChinese : .simplifiedChinese
    }

    /// `english` in the current language; `english` itself where there is no translation.
    fileprivate func text(_ english: String, context: String) -> String {
        let key = Translation.key(english, context: context)
        switch language {
        case .english: return english
        case .simplifiedChinese: return Translations.simplified[key] ?? english
        case .traditionalChinese: return Translations.traditional[key] ?? english
        }
    }
}

/// The text `english` in the app's current language, with `{0}`, `{1}` ... replaced by
/// `values` in that order. Write the English text as ONE string literal: the check that
/// every text has both translations reads the literals after `tr(`.
func tr(_ english: String, _ values: CustomStringConvertible...) -> String {
    let text = Localizer.shared.text(english, context: "")
    return values.isEmpty ? text : filled(text, with: values.map { $0.description })
}

/// For an English word that needs two different translations: `context` says which one
/// ("size" is a picture's dimensions in one place and a lever in another).
func tr(_ english: String, context: String) -> String {
    Localizer.shared.text(english, context: context)
}

/// Marks a string literal that is kept in a constant and shown later through
/// `tr(variable)`, so the translation check finds it. Returns it unchanged.
func trKey(_ english: String) -> String { english }

/// `template` with each `{n}` replaced by `values[n]`, in one pass: a value that itself
/// contains braces is not read again. A `{...}` that names no value stays as written.
private func filled(_ template: String, with values: [String]) -> String {
    var out = ""
    var rest = Substring(template)
    while let open = rest.firstIndex(of: "{") {
        out.append(contentsOf: rest[..<open])
        let afterOpen = rest.index(after: open)
        if let close = rest[afterOpen...].firstIndex(of: "}"),
           let n = Int(rest[afterOpen..<close]), values.indices.contains(n) {
            out.append(values[n])
            rest = rest[rest.index(after: close)...]
        } else {
            out.append("{")
            rest = rest[afterOpen...]
        }
    }
    out.append(contentsOf: rest)
    return out
}

/// One text in the three languages. `context` is empty except where one English word has
/// two translations (`tr(_:context:)`).
struct Translation {
    let english: String
    let simplified: String
    let traditional: String
    let context: String

    init(_ english: String, _ simplified: String, _ traditional: String, context: String = "") {
        self.english = english
        self.simplified = simplified
        self.traditional = traditional
        self.context = context
    }

    /// What a table is looked up by: the English text, with its context where it has one.
    static func key(_ english: String, context: String) -> String {
        context.isEmpty ? english : english + "\u{1F}" + context
    }
}
