#!/usr/bin/env python3
"""The iOS app's translations: check them, hand them to a reviewer, take the review back.

    python3 ios/tools/translations.py check
        Every text the code passes to tr() / trKey() has a row in App/Translations.swift and
        every row is used; a row's three texts hold the same {n} placeholders and each call
        hands over as many values; every View that calls tr() observes the Localizer; every
        Picker whose options have translated names is built again for a new language.
        Also lists, to be looked at by hand, tr() calls whose text is not a literal and
        interface literals that do not go through tr().

    python3 ios/tools/translations.py export [review.md]
        A Markdown file with one three-column table per screen: English, Simplified
        Chinese, Traditional Chinese. Without a file name it is printed.

    python3 ios/tools/translations.py import review.md
        Reads an amended review file back: the two Chinese columns replace the rows of
        Translations.swift and the two permission prompts in App/zh-*.lproj. The English
        column is the key and must come back as it went out: a text that is to be
        reworded, added or dropped in English is changed in the code first (the tr() call
        and its row), then the file is read. A row whose first cell begins with /// is a
        note to the developer; it is printed and never read into the app.

Nothing here is needed to build the app. Python 3.8 or later, no packages.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(os.path.dirname(HERE), "App")
TABLE = os.path.join(APP, "Translations.swift")
PROMPTS = {"hans": os.path.join(APP, "zh-Hans.lproj", "InfoPlist.strings"),
           "hant": os.path.join(APP, "zh-Hant.lproj", "InfoPlist.strings"),
           "en": os.path.join(APP, "en.lproj", "InfoPlist.strings")}
PROMPT_HEADING = "iOS permission prompts"
NOTE_ROW = "///"        # a review-file row that starts with this is a note to the developer

LIT = r'"((?:[^"\\]|\\.)*)"'
ROW = re.compile(r'^(\s*)T\(' + LIT + r', ' + LIT + r', ' + LIT + r'(?:, context: "([^"]*)")?\),\s*$')
MARK = re.compile(r'^\s*// MARK: (.+?)\s*$')
PLACEHOLDER = re.compile(r'\{(\d+)\}')
STRINGS_ROW = re.compile(r'^\s*"([^"]+)"\s*=\s*' + LIT + r'\s*;\s*$')

# What a reviewer cannot tell from a short text alone. Shown in the English column of the
# review file, after the text; never part of the key.
NOTES = {
    ("Live", ""): "tab title: detection on the camera picture",
    ("Photo", ""): "tab title: detection on picked photos",
    ("Bench", ""): "tab title: benchmark",
    ("Compute", ""): "the compute unit picker (Auto / GPU / CPU), and a section title in Settings",
    ("Auto", ""): "compute unit: Core ML decides where the model runs",
    ("Pose", ""): "model type: pose estimation",
    ("System", ""): "language choice: follow the device's language",
    ("Built on", ""): "heading over the two projects the app is based on",
    ("Pipeline", ""): "model attribute: how the model is run",
    ("Operations", ""): "model attribute: number of operations in the model",
    ("Origin", ""): "model attribute: bundled with the app, or imported",
    ("Source", ""): "model attribute: what the model was exported from",
    ("Kpt score", ""): "model attribute: keypoint score",
    ("Size", ""): "model attribute: size on disk",
    ("camera", ""): "stats row: frames per second the camera delivers",
    ("loop", ""): "stats row: rate of the detection loop",
    ("frame age", ""): "stats row: how old a frame is when it arrives",
    ("dets", ""): "under the number of detected objects; very short",
    ("mask", ""): "stats row: time to draw the segmentation masks",
    ("model 1", ""): "stats row of a two-model pipeline: time of the first model (it finds the people)",
    ("model 2", ""): "stats row of a two-model pipeline: time of the second model (the pose of each person)",
    ("end to end", ""): "stats row: total time for one image",
    ("Style", ""): "picker: how boxes are drawn",
    ("Solid", ""): "box style",
    ("Neon", ""): "box style",
    ("Chip", ""): "box style: label on a small plate",
    ("Minimal", ""): "box style",
    ("Overlay", ""): "picker: what is drawn for a segmentation model",
    ("Pose overlay", ""): "picker: what is drawn for a pose model",
    ("Masks", ""): "overlay choice",
    ("Boxes", ""): "overlay choice: bounding boxes",
    ("Both", ""): "overlay choice",
    ("Skeleton", ""): "overlay choice",
    ("box", ""): "lever: box line thickness and label size; at most 3 characters",
    ("size", ""): "lever: skeleton size; at most 3 characters",
    ("size", "image"): "stats row: the image's dimensions in pixels",
    ("image", ""): "stats row: which image of the batch is open",
    ("images", ""): "stats row: number of images in the batch",
    ("file", ""): "stats row: file name",
    ("type", ""): "stats row: file type",
    ("wall", ""): "stats row: wall-clock time of the whole batch",
    ("total dets", ""): "stats row: detections in the whole batch",
    ("avg e2e", ""): "stats row: average end-to-end time",
    ("Loading / iCloud", ""): "progress bar: photos are loading, some from iCloud",
    ("Inference", ""): "progress bar: the model is running",
    ("Deselect", ""): "button: clears the selection",
    ("Remove", ""): "button: takes photos out of the batch",
    ("Cold run", ""): "benchmark mode: a short run on a cool device",
    ("Sustained", ""): "benchmark mode: a long run that shows thermal throttling",
    ("on", ""): "between two menus: \"All 5 models on All 3 units\"",
    ("Unit", ""): "the compute unit picker in Bench",
    ("Timed iters: {0}", ""): "number of timed iterations",
    ("Warmup: {0}", ""): "number of warm-up iterations",
    ("cold {0}", ""): "the cold baseline, in ms",
    ("FASTEST", ""): "heading of the card with the fastest result",
    ("fastest {0}", ""): "tag on a result card: the fastest compute unit",
    ("sustained", ""): "row name: latency in a sustained run",
    ("throttle +{0}%", ""): "how much slower the device got from heat",
    ("Time", ""): "sort order of the history",
    ("Name", ""): "sort order of the history, and the field for a run's name",
}


def fail_exit(msg):
    print("error:", msg)
    sys.exit(1)


def read_table():
    """-> (sections, rows). rows: dicts with en, hans, hant, ctx, line, section."""
    rows, section = [], "Shared"
    for no, line in enumerate(open(TABLE, encoding="utf8"), 1):
        mark = MARK.match(line)
        if mark:
            section = mark.group(1)
            continue
        if re.match(r'^\s*T\(', line):
            m = ROW.match(line)
            if not m:
                fail_exit(f"Translations.swift:{no}: a row is not in the one-line form")
            rows.append(dict(indent=m.group(1), en=m.group(2), hans=m.group(3), hant=m.group(4),
                             ctx=m.group(5) or "", line=no, section=section))
    return rows


def read_strings(path):
    out = {}
    for line in open(path, encoding="utf8"):
        m = STRINGS_ROW.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def placeholders(text):
    return sorted(set(PLACEHOLDER.findall(text)))


# ---------------------------------------------------------------- check

def strip_comment(line):
    out, i, instr = [], 0, False
    while i < len(line):
        c = line[i]
        if instr:
            out.append(c)
            if c == '\\' and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                instr = False
        else:
            if c == '/' and line[i:i + 2] == '//':
                break
            out.append(c)
            if c == '"':
                instr = True
        i += 1
    return ''.join(out)


def call_args(text, start):
    """text[start] is the '(' of a call: its top-level arguments, or None if unbalanced."""
    depth, i, instr, args, cur = 0, start, False, [], []
    while i < len(text):
        c = text[i]
        if instr:
            cur.append(c)
            if c == '\\':
                cur.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                instr = False
        elif c == '"':
            instr = True
            cur.append(c)
        elif c in '([{':
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                args.append(''.join(cur).strip())
                return args
            cur.append(c)
        elif c == ',' and depth == 1:
            args.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    return None


def close_of(text, i):
    """text[i] opens a bracket: the index of the bracket that closes it, or None."""
    depth, instr = 0, False
    while i < len(text):
        c = text[i]
        if instr:
            if c == '\\':
                i += 2
                continue
            if c == '"':
                instr = False
        elif c == '"':
            instr = True
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def picker_parts(code, start):
    """code[start] is the '(' of `Picker(`: -> (its options, the modifiers that follow it), or None."""
    def blank(j):
        while j < len(code) and code[j].isspace():
            j += 1
        return j
    end = close_of(code, start)
    if end is None:
        return None
    j, options = blank(end + 1), ""
    if code[j:j + 1] == "{":
        end = close_of(code, j)
        if end is None:
            return None
        options, j = code[j:end + 1], blank(end + 1)
        label = re.match(r'label:\s*\{', code[j:])
        if label:
            end = close_of(code, j + label.end() - 1)
            if end is None:
                return None
            j = blank(end + 1)
    first = j
    while code[j:j + 1] == ".":
        name = re.match(r'\.\w+', code[j:])
        if not name:
            break
        j += name.end()
        if code[j:j + 1] == "(":
            end = close_of(code, j)
            if end is None:
                return None
            j = end + 1
        j = blank(j)
    return options, code[first:j]


UI = re.compile(
    r'\b(Text|Label|Button|Toggle|Picker|TextField|Section|Menu|Link|ContentUnavailableView|ProgressView'
    r'|DisclosureGroup|LabeledContent)\(\s*"([^"]*[A-Za-z][^"]*)"'
    r'|\.(navigationTitle|alert|accessibilityLabel|accessibilityHint|confirmationDialog|help)\(\s*"([^"]*)"'
    r'|prompt:\s*"([^"]*)"|(title|name|message|description):\s*"([^"]*[A-Za-z]{3}[^"]*)"')


def check():
    problems = []
    rows = read_table()
    seen = {}
    for r in rows:
        key = (r["en"], r["ctx"])
        where = f"Translations.swift:{r['line']}"
        if key in seen:
            problems.append(f"{where}: the same text as line {seen[key]}: {r['en']!r}")
        seen[key] = r["line"]
        if not (placeholders(r["en"]) == placeholders(r["hans"]) == placeholders(r["hant"])):
            problems.append(f"{where}: the three texts do not hold the same placeholders: {r['en'][:60]!r}")
        for name in ("hans", "hant"):
            if not r[name].strip():
                problems.append(f"{where}: an empty translation")
        if '\\' in r["en"] + r["hans"] + r["hant"]:
            problems.append(f"{where}: a backslash (write the character itself)")
    table = set(seen)

    used, dynamic, bypass = {}, [], []
    for path in sorted(glob.glob(os.path.join(APP, "*.swift"))):
        name = os.path.basename(path)
        if name == "Translations.swift":
            continue
        code = "\n".join(strip_comment(l) for l in open(path, encoding="utf8").read().split("\n"))
        for m in re.finditer(r'\b(tr|trKey)\(', code):
            if re.search(r'\bfunc\s+$', code[max(0, m.start() - 8):m.start()]):
                continue                                    # the declaration itself
            line = code.count("\n", 0, m.start()) + 1
            args = call_args(code, m.end() - 1)
            if args is None:
                problems.append(f"{name}:{line}: unbalanced call")
                continue
            lit = re.fullmatch(LIT, args[0]) if args else None
            if not lit:
                dynamic.append(f"{name}:{line}: {m.group(1)}({args[0] if args else ''})")
                continue
            en, ctx, values = lit.group(1), "", args[1:]
            if values and values[-1].startswith("context:"):
                c = re.fullmatch(r'context:\s*"([^"]*)"', values[-1])
                if not c:
                    problems.append(f"{name}:{line}: the context is not a literal")
                    continue
                ctx, values = c.group(1), values[:-1]
            used.setdefault((en, ctx), []).append(f"{name}:{line}")
            if m.group(1) == "tr":
                idx = sorted(int(x) for x in placeholders(en))
                if idx != list(range(len(idx))):
                    problems.append(f"{name}:{line}: placeholders are not {{0}}..{{n}}: {en[:60]!r}")
                if len(values) != len(idx):
                    problems.append(f"{name}:{line}: {len(idx)} placeholders but {len(values)} values: {en[:60]!r}")
        for m in re.finditer(r'\bPicker\(', code):
            line = code.count("\n", 0, m.start()) + 1
            parts = picker_parts(code, m.end() - 1)
            if parts is None:
                problems.append(f"{name}:{line}: unbalanced Picker")
            elif (re.search(r'\.label\b|\.displayName\b|\btr\(', parts[0])
                  and ".id(localizer.language)" not in parts[1]):
                problems.append(f"{name}:{line}: a Picker whose options have translated names "
                                "lacks .id(localizer.language)")
        for m in UI.finditer(code):
            line = code.count("\n", 0, m.start()) + 1
            bypass.append(f"{name}:{line}: {code[m.start():m.end()][:90]}")
        for m in re.finditer(r'^(?:private |fileprivate )?struct (\w+)(?:<[^>]*>)?\s*:\s*[^{\n]*\bView\b[^{\n]*\{',
                             code, re.M):
            depth, i = 0, m.end() - 1
            while i < len(code):
                if code[i] == '{':
                    depth += 1
                elif code[i] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            body = code[m.start():i]
            if re.search(r'\btr\(', body) and "Localizer.shared" not in body:
                problems.append(f"{name}: struct {m.group(1)} calls tr() but does not observe the Localizer")

    for key, where in sorted(used.items()):
        if key not in table:
            ctx = f" [context: {key[1]}]" if key[1] else ""
            problems.append(f"{where[0]}: no translation for {key[0]!r}{ctx}")
    for r in rows:
        if (r["en"], r["ctx"]) not in used:
            problems.append(f"Translations.swift:{r['line']}: used nowhere in the code: {r['en'][:70]!r}")

    en_prompts = read_strings(PROMPTS["en"])
    for lang in ("hans", "hant"):
        got = read_strings(PROMPTS[lang])
        if set(got) != set(en_prompts):
            problems.append(f"{os.path.relpath(PROMPTS[lang], os.path.dirname(APP))}: keys differ from en.lproj")
    spec = open(os.path.join(os.path.dirname(APP), "project.yml"), encoding="utf8").read()
    for key, text in en_prompts.items():
        if f'{key}: "{text}"' not in spec:
            problems.append(f"en.lproj/InfoPlist.strings: {key} is not the text in project.yml")

    print(f"{len(rows)} rows, {len(used)} texts in the code, {sum(len(w) for w in used.values())} call sites, "
          f"{len(en_prompts)} permission prompts")
    print("tr() with a text that is not a literal (look at these by hand):")
    for d in dynamic:
        print("   ", d)
    print("interface literals that do not go through tr() (look at these by hand):")
    for b in bypass:
        print("   ", b)
    for p in problems:
        print("FAIL", p)
    print("result:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


# ---------------------------------------------------------------- export

def cell(text):
    return text.replace("|", "\\|")


def english_cell(r):
    out = cell(r["en"])
    if r["ctx"]:
        out += f" [context: {r['ctx']}]"
    note = NOTES.get((r["en"], r["ctx"]))
    if note:
        out += f" *(note: {cell(note)})*"
    return out


def export(target):
    rows = read_table()
    out = ["# YOLO-Master Edge for iOS: translations for review",
           "",
           "Every text of the app in English, Simplified Chinese and Traditional Chinese.",
           "",
           "How to review",
           "",
           "- Change the second and third column freely. Leave the first column as it is: the app finds a translation by its English text.",
           "- `{0}`, `{1}` and so on are values the app fills in (a number, a file name, a model name). Keep every one of them. They may move within the sentence.",
           "- These stay in English on purpose: CPU, GPU, Neural Engine, Core ML, NMS, IoU, FPS, ms, e2e, HUD, model and project names, file extensions, licence names.",
           "- Simplified Chinese puts “ ” and Traditional Chinese 「 」 around the name of a button, a tab or an app.",
           "- Traditional Chinese is one translation for Taiwan, Hong Kong and Macau.",
           "- Do not use the straight double quote `\"` or a backslash in a translation.",
           "- An *(note: ...)* after an English text says where it appears. It is not shown in the app.",
           "- To say something to the developer (an English text that should change, a text to add or to drop), add a row whose first cell begins with `///`. Such a row is read as a note and never shown in the app.",
           "- The licence notice in the grey box in Settings is not in this file: it stays in English.",
           ""]
    section = None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            out += ["", f"## {section}", "", "| English | 简体中文 | 繁體中文 |", "|---|---|---|"]
        out.append(f"| {english_cell(r)} | {cell(r['hans'])} | {cell(r['hant'])} |")
    en, hans, hant = (read_strings(PROMPTS[k]) for k in ("en", "hans", "hant"))
    out += ["", f"## {PROMPT_HEADING}", "",
            "Shown by iOS itself when the app first asks for the camera, and when it first saves a picture.", "",
            "| English | 简体中文 | 繁體中文 |", "|---|---|---|"]
    for key in en:
        out.append(f"| {cell(en[key])} | {cell(hans.get(key, ''))} | {cell(hant.get(key, ''))} |")
    text = "\n".join(out) + "\n"
    if target:
        open(target, "w", encoding="utf8").write(text)
        print(f"{target}: {len(rows)} texts and {len(en)} permission prompts")
    else:
        sys.stdout.write(text)
    return 0


# ---------------------------------------------------------------- import

def split_row(line):
    """A Markdown table row -> its cells, with \\| read as |. None if it is not a row."""
    s = line.strip()
    if not (s.startswith("|") and s.endswith("|")):
        return None
    cells, cur, i = [], [], 1
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if s[i] == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(s[i])
        i += 1
    return cells


def import_review(source):
    rows = read_table()
    by_key = {(r["en"], r["ctx"]): r for r in rows}
    en_prompts = read_strings(PROMPTS["en"])
    prompt_key = {text: key for key, text in en_prompts.items()}
    new_prompts = {"hans": dict(read_strings(PROMPTS["hans"])), "hant": dict(read_strings(PROMPTS["hant"]))}
    problems, changed, seen, in_prompts, notes = [], [], set(), False, []
    for no, line in enumerate(open(source, encoding="utf8"), 1):
        if line.startswith("## "):
            in_prompts = line[3:].strip() == PROMPT_HEADING
            continue
        cells = split_row(line)
        if cells and cells[0].startswith(NOTE_ROW):
            notes.append(f"{source}:{no}: {' | '.join(c for c in cells if c)[len(NOTE_ROW):].strip()}")
            continue
        if cells is None or len(cells) != 3 or cells[0] == "English" or set(cells[0]) <= set("-: "):
            continue
        en, hans, hant = cells
        en = re.sub(r'\s*\*\(note: .*\)\*$', '', en)
        for text in (hans, hant):
            if '"' in text or '\\' in text:
                problems.append(f"{source}:{no}: a straight double quote or a backslash in a translation")
        if not hans or not hant:
            problems.append(f"{source}:{no}: an empty translation")
            continue
        if in_prompts:
            key = prompt_key.get(en)
            if key is None:
                problems.append(f"{source}:{no}: not one of the permission prompts: {en[:60]!r}")
                continue
            for lang, text in (("hans", hans), ("hant", hant)):
                if new_prompts[lang].get(key) != text:
                    changed.append(f"prompt {key} ({lang})")
                new_prompts[lang][key] = text
            continue
        ctx = ""
        m = re.search(r' \[context: ([^\]]+)\]$', en)
        if m:
            ctx, en = m.group(1), en[:m.start()]
        r = by_key.get((en, ctx))
        if r is None:
            problems.append(f"{source}:{no}: the English text is not one of the app's: {en[:70]!r}")
            continue
        if not (placeholders(en) == placeholders(hans) == placeholders(hant)):
            problems.append(f"{source}:{no}: placeholders differ from the English text: {en[:60]!r}")
            continue
        seen.add((en, ctx))
        if (r["hans"], r["hant"]) != (hans, hant):
            changed.append(en[:70])
            r["hans"], r["hant"] = hans, hant
    missing = [r["en"] for r in rows if (r["en"], r["ctx"]) not in seen]
    if notes:
        print(f"{len(notes)} note(s) to the developer in the file (not read into the app):")
        for n in notes:
            print("   ", n)
    if problems:
        for p in problems:
            print("FAIL", p)
        print("nothing was changed")
        return 1
    lines = open(TABLE, encoding="utf8").read().split("\n")
    for r in rows:
        ctx = f', context: "{r["ctx"]}"' if r["ctx"] else ""
        lines[r["line"] - 1] = f'{r["indent"]}T("{r["en"]}", "{r["hans"]}", "{r["hant"]}"{ctx}),'
    open(TABLE, "w", encoding="utf8").write("\n".join(lines))
    for lang in ("hans", "hant"):
        text = open(PROMPTS[lang], encoding="utf8").read().split("\n")
        for i, line in enumerate(text):
            m = STRINGS_ROW.match(line)
            if m and m.group(1) in new_prompts[lang]:
                text[i] = f'"{m.group(1)}" = "{new_prompts[lang][m.group(1)]}";'
        open(PROMPTS[lang], "w", encoding="utf8").write("\n".join(text))
    print(f"{len(changed)} text(s) changed" + (":" if changed else ""))
    for c in changed:
        print("   ", c)
    if missing:
        print(f"{len(missing)} text(s) of the app are not in the review file and were left as they are:")
        for en in missing:
            print("   ", en[:70])
    return 0


def main():
    args = sys.argv[1:]
    if args[:1] == ["check"] and len(args) == 1:
        return check()
    if args[:1] == ["export"] and len(args) <= 2:
        return export(args[1] if len(args) == 2 else None)
    if args[:1] == ["import"] and len(args) == 2:
        return import_review(args[1])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
