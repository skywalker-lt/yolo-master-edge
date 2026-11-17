// The translations of the app's text: English (the key, exactly as written in the code),
// Simplified Chinese, Traditional Chinese. See Localization.swift for how they are used.
//
// Rules of this file
//   - ONE row to a line: T("English", "简体中文", "繁體中文"). The review table handed to
//     translators is generated from these rows and read back into them.
//   - The English text must match the text in the code to the character. A row whose English
//     matches nothing is dead; a text in the code with no row here is shown in English.
//   - `{0}`, `{1}` ... are values the app fills in. Keep every one of them; their order may
//     change.
//   - Not translated: CPU, GPU, Neural Engine, Core ML, NMS, IoU, FPS, ms, e2e, HUD, model
//     and project names, file extensions, licence names.
//   - Simplified Chinese uses “ ” and Traditional Chinese 「 」 around the name of a button,
//     a tab or an app. A name in quotes is the name on the screen: when a label's
//     translation changes, change the sentences that quote it too.
//   - Traditional Chinese is one table for Taiwan, Hong Kong and Macau. Both Chinese columns
//     were amended by a native reviewer (1.2.1); a new row takes its words from the rows
//     already here (目标检测 / 物件偵測, 相片 for a picked photo, 「照片」 and 照片圖庫 for
//     Apple's Photos app and library, 「檔案」 for Apple's Files app).
//   - Chinese and Latin words are set apart by a space: 本 App, 使用 CPU 推理.
import Foundation

private typealias T = Translation

enum Translations {
    static let simplified: [String: String] = table(\.simplified)
    static let traditional: [String: String] = table(\.traditional)

    /// A later row with the same English text and context replaces an earlier one.
    private static func table(_ column: KeyPath<Translation, String>) -> [String: String] {
        var out: [String: String] = [:]
        for row in general + settings + live + photo + bench {
            out[Translation.key(row.english, context: row.context)] = row[keyPath: column]
        }
        return out
    }

    // MARK: Shared by several screens

    private static let general: [Translation] = [
        T("OK", "好", "好"),
        T("Cancel", "取消", "取消"),
        T("Save", "保存", "儲存"),
        T("Done", "完成", "完成"),
        T("Open Settings", "打开设置", "打開設定"),
        T("Live", "实时", "即時"),
        T("Photo", "照片", "相片"),
        T("Bench", "基准测试", "基準測試"),
        T("Settings", "设置", "設定"),
        T("Model", "模型", "模型"),
        T("Models", "模型", "模型"),
        T("Compute", "计算单元", "運算單元"),
        T("Auto", "自动", "自動"),
        T("Detection", "检测", "偵測"),
        T("Segmentation", "分割", "分割"),
        T("Pose", "姿势", "姿態"),
        T("Whole-body pose", "全身姿势", "全身姿態"),
        T("Loading the model ({0})", "正在加载模型（{0}）", "正在載入模型（{0}）"),
        T("Select all", "全选", "全選"),
        T("Export failed", "导出失败", "導出失敗"),
        T("The image could not be turned into the model's input.", "无法将图片转换为模型的输入。", "無法將圖片轉換為模型的輸入。"),
        T("The model did not return a detection tensor this app can read.", "模型没有返回能够读取的张量。", "模型沒有傳回能夠讀取的張量。"),
        T("This pose model works on one person at a time and needs a detection model with a “person” class next to it (for example yolo26n). Add one first.", "该姿势模型一次只能识别到一个人，需要搭配一个包含“person”类别的目标检测模型（如 yolo26n）。请先添加该模型。", "此姿態模型一次僅可偵測到一個人，需搭配一個包含「person」類別的物件偵測模型（如 yolo26n）。請先添加該模型。"),
    ]

    // MARK: Settings

    private static let settings: [Translation] = [
        T("Language", "语言", "語言"),
        T("System", "系统", "系統"),
        T("System uses the language of this {0}.", "“系统”会使用此 {0} 的语言。", "「系統」會使用此 {0} 的語言。"),
        T("Erase all history?", "删除全部历史记录？", "刪除所有歷史紀錄？"),
        T("Erase", "删除", "刪除"),
        T("This permanently deletes all {0} saved benchmark runs.", "这将永久删除全部 {0} 条已保存的基准测试记录。", "這將永久刪除全部 {0} 筆已儲存的基準測試紀錄。"),
        T("Import failed", "导入失败", "加載失敗"),
        T("YOLO-Master Edge runs YOLO-Master and other modern object detection models completely on your device, using Core ML for acceleration. It is one of the best open-source on-device computer vision tools for iOS. You can run and benchmark your own models with the “Load custom Core ML model” feature.", "YOLO-Master Edge 在你的设备本地通过 Core ML 运行并加速 YOLO-Master 及其它目标检测模型，是目前 iOS 上功能最强大的开源端侧计算机视觉工具之一。你可以随时使用下方的“加载自定义 Core ML 模型”功能导入并测试你自己训练的模型。", "YOLO-Master Edge 在你的設備本地透過 Core ML 運行並加速 YOLO-Master 及其它物件偵測模型，是目前 iOS 上最強大的開源裝置端 CV 工具之一。你可以隨時通過下方「載入自訂 Core ML 模型」功能載入並測試你自己訓練的模型。"),
        T("YOLO-Master extends the real-time YOLO detector with Mixture-of-Experts (MoE) routing. Instead of one dense network, lightweight expert branches specialize on different feature patterns, and a learned router activates only the most relevant experts for each image.", "YOLO-Master 在 YOLO 实时检测框架的基础上引入了混合专家（Mixture-of-Experts / MoE）路由机制，模型不再依赖单一的稠密计算网络，而是由多个轻量级专家分支分别计算不同的特征，并由一个可训练的路由决定每张图像只激活特定的专家。", "YOLO-Master 在 YOLO 即時物件偵測架構的基礎上引入了混合專家（Mixture-of-Experts，MoE）路由機制。模型不再依賴單一稠密計算網路，而是由多個輕量級專家分支分別專注於不同的特徵，並由一個可訓練的路由決定每張影像只啟用特定的專家組合。"),
        T("Paper", "论文", "論文"),
        T("Model Repo", "模型仓库", "模型儲存庫"),
        T("App Repo", "本应用仓库", "本 App 儲存庫"),
        T("Version {0}", "版本 {0}", "版本 {0}"),
        T("Version {0} ({1})", "版本 {0} ({1})", "版本 {0} ({1})"),
        T("On-device multi-task detection with Core ML.", "基于 Core ML 的端侧多任务检测。", "以 Core ML 實現的裝置端多功能偵測。"),
        T("The source code of this app is licensed under AGPL-3.0. The models it bundles come from the projects below and remain under their own terms: YOLO-Master and Ultralytics under AGPL-3.0, MMPose under Apache-2.0. coremltools is licensed under BSD-3-Clause.", "本应用的源码以 AGPL-3.0 许可证发布并开源。应用内置的模型源自以下项目并继续适用这些项目各自的条款：YOLO-Master 和 Ultralytics 为 AGPL-3.0 许可证，MMPose 为 Apache-2.0 许可证。coremltools 为 BSD-3-Clause 许可证。", "本 App 的原始碼以 AGPL-3.0 授權條款釋出並開源。App 內建的模型來自下列專案，並繼續適用這些專案各自的條款：YOLO-Master 與 Ultralytics 為 AGPL-3.0，MMPose 為 Apache-2.0。coremltools 為 BSD-3-Clause 授權條款。"),
        T("This app is non-commercial. The authors release it for research and personal use only.", "本应用不可直接商用。作者发布本应用仅供科研和个人体验。", "本 App 不允許直接的商業用途。作者釋出本 App 僅供科學研究與個人體驗。"),
        T("This notice is about the app as distributed, with the models it bundles. It does not change the AGPL-3.0 terms of the source code.", "本声明针对的是发布的应用本身，不改变源代码所适用的 AGPL-3.0 条款。", "本聲明針對釋出的 App 本身，不改變原始碼所適用的 AGPL-3.0 條款。"),
        T("Built on", "基于以下项目构建", "基於下列專案構建"),
        T("Acknowledgements", "致谢", "致謝"),
        T("Licenses & Acknowledgements", "许可与致谢", "授權與致謝"),
        T("YOLO-Master Edge (this app)", "YOLO-Master Edge（本应用）", "YOLO-Master Edge（本 App）"),
        T("The MoE YOLO detector family this app accelerates.", "本应用所优化的 MoE 目标检测模型。", "本 App 所加速的 MoE 物件偵測模型。"),
        T("The source code of this app and the Core ML export toolchain.", "本应用的源码以及 Core ML 导出工具链。", "本 App 的原始碼以及 Core ML 匯出工具鏈。"),
        T("The YOLO training and inference framework YOLO-Master builds on.", "YOLO-Master 所基于的 YOLO 训练与推理框架。", "YOLO-Master 所採用的 YOLO 訓練與推論框架。"),
        T("The RTMW whole-body pose model bundled with this app.", "本应用内置的 RTMW 全身姿势模型。", "本 App 內建的 RTMW 全身姿態模型。"),
        T("Model conversion and the on-device inference runtime.", "模型转换以及端侧推理 Runtime。", "模型轉換以及裝置端推論 Runtime。"),
        T("YOLO-Master Edge runs entirely on your device.", "YOLO-Master Edge 完全在你的本地运行。", "YOLO-Master Edge 完全在你的裝置上執行。"),
        T("Camera", "相机", "相機"),
        T("Live on-device detection. Frames are processed in memory and never stored or transmitted.", "用于实时端侧检测。画面只在内存中处理，不会被存储或传输。", "用於即時裝置端偵測。畫面只在記憶體中處理，不會被儲存或傳輸。"),
        T("Photo Library", "照片图库", "照片圖庫"),
        T("Reads only the images you pick for the Photo tab, and saves detection frames you explicitly export.", "仅读取你为“照片”标签页选取的图片，并保存你导出的检测结果图。", "僅讀取你為「相片」分頁選取的圖片，並儲存你匯出的偵測結果圖。"),
        T("Privacy & Security", "隐私与安全", "隱私與安全性"),
        T("Privacy Policy", "隐私政策", "隱私權政策"),
        T("Allow CPU inference", "使用 CPU 推理", "啟用 CPU 運行"),
        T("Inference on CPU is slow and heats up the device rapidly. Some models may not run reliably on CPU.", "CPU 推理速度相对较慢并会使设备迅速发热。部分特定的模型在 CPU 上可能无法稳定运行。", "CPU 速度較慢並會使裝置迅速發熱。部分模型在 CPU 上可能無法穩定執行。"),
        T("Model gallery", "模型库", "模型庫"),
        T("Type", "类型", "類型"),
        T("{0} (going by the file name: the model has not been compiled yet)", "{0}（根据文件名推断：模型尚未编译）", "{0}（依檔名推斷：模型尚未編譯）"),
        T("Keypoints", "关键点", "關鍵點"),
        T("133 per person: 17 body, 6 feet, 68 face, 42 hands", "每人 133 个：身体 17、足部 6、面部 68、手部 42", "每人 133 個：身體 17、足部 6、臉部 68、手部 42"),
        T("{0} per person", "每人 {0} 个", "每人 {0} 個"),
        T("Pipeline", "流程", "流程"),
        T("Top-down: a detection model finds the people, this model runs once per person", "自上而下：先由目标检测模型找出人物，再由本模型对每个人进行检测", "由上而下：先由物件偵測模型找出人物，本模型再對每個人執行一次"),
        T("One pass, NMS-free output", "单次推理，输出无需 NMS", "單次推論，輸出無需 NMS"),
        T("One pass, NMS in the app", "单次推理，由本应用执行 NMS", "單次推論，由本 App 執行 NMS"),
        T("Classes", "类别", "類別"),
        T("{0}: {1} and {2} more", "{0}：{1} 等，另有 {2} 个", "{0}：{1} 等，另有 {2} 個"),
        T("Input", "输入", "輸入"),
        T("Output", "输出", "輸出"),
        T("Precision", "精度", "精度"),
        T("FP16: 16-bit float weights and activations", "FP16：权重与激活值均为 16 位浮点", "FP16：權重與激活值皆為 16 位元浮點"),
        T("W8A16: 8-bit weights, 16-bit float activations", "W8A16：8 位权重，16 位浮点激活值", "W8A16：8 位元權重，16 位元浮點激活值"),
        T("W8A8: 8-bit weights and activations", "W8A8：权重与激活值均为 8 位", "W8A8：權重與激活值皆為 8 位元"),
        T("{0}. Stored as {1}", "{0}。存储格式为 {1}", "{0}。儲存格式為 {1}"),
        T("Format", "格式", "格式"),
        T("{0}, specification {1}", "{0}，规范版本 {1}", "{0}，規格版本 {1}"),
        T("{0}, iOS {1} or later", "{0}，需要 iOS {1} 或更高版本", "{0}，需要 iOS {1} 或以上版本"),
        T("Operations", "算子数", "運算數"),
        T("Size", "大小", "大小"),
        T("Origin", "来源", "來源"),
        T("Imported in Settings", "在“设置”中导入", "在「設定」中載入"),
        T("Bundled with the app", "应用内置", "App 內建"),
        T("Kpt score", "关键点分数", "關鍵點分數"),
        T("drawn from {0}", "从 {0} 起绘制", "從 {0} 起繪製"),
        T("Source", "源模型", "來源模型"),
        T("About", "简介", "簡介"),
        T("Author", "作者", "作者"),
        T("License", "许可证", "授權條款"),
        T("Version", "版本", "版本"),
        T("File", "文件", "檔案"),
        T("shape not declared", "未声明形状", "未宣告形狀"),
        T("Load custom Core ML model", "加载自定义 Core ML 模型", "載入自訂 Core ML 模型"),
        T("Load models from a folder", "从文件夹加载模型", "從檔案夾載入模型"),
        T("Checking the model...", "正在检查模型…", "正在檢查模型…"),
        T("Custom models", "自定义模型", "自訂模型"),
        T("Import a .mlpackage, .mlmodelc, or .mlmodel. Use Load models from a folder if Files displays a model package as a folder.", "可导入 .mlpackage、.mlmodelc 或 .mlmodel。如果“文件”App 将模型识别为文件夹，请使用“从文件夹加载模型”。", "可匯入 .mlpackage、.mlmodelc 或 .mlmodel。如果「檔案」App 把模型套件顯示為檔案夾，請使用「從檔案夾載入模型」。"),
        T("Erase all benchmark history", "删除全部基准测试历史", "刪除所有基準測試歷史"),
        T("{0} saved run.", "已保存 {0} 条记录。", "已儲存 {0} 筆紀錄。"),
        T("{0} saved runs.", "已保存 {0} 条记录。", "已儲存 {0} 筆紀錄。"),
        T("Report an issue", "报告问题", "回報問題"),
        T("Opens a new issue on GitHub in your browser.", "在浏览器中打开 GitHub 的新建 Issue 页面。", "在瀏覽器中開啟 GitHub 的新增 Issue 頁面。"),
        T("{0} is not a Core ML model and contains none (.mlpackage, .mlmodelc, .mlmodel).", "{0} 不是 Core ML 模型，其中也不包含任何模型（.mlpackage、.mlmodelc、.mlmodel）。", "{0} 不是 Core ML 模型，其中也不包含任何模型（.mlpackage、.mlmodelc、.mlmodel）。"),
        T("{0} was not imported. {1}", "{0} 未导入。{1}", "{0} 未加載。{1}"),
    ]

    // MARK: Live, the stats HUD and the tuning panel

    private static let live: [Translation] = [
        T("Initializing...", "正在初始化…", "正在初始化…"),
        T("No models bundled", "未内置模型", "未內建模型"),
        T("Copy .mlpackage files into ios/Models/, re-run xcodegen, rebuild.", "请将 .mlpackage 文件复制到 ios/Models/，重新运行 xcodegen 并重新构建。", "請將 .mlpackage 檔案複製到 ios/Models/，重新執行 xcodegen 並重新構建。"),
        T("Camera access required", "需要相机访问权限", "需要相機取用權限"),
        T("Allow camera access to run live detection", "请允许访问相机以进行实时检测", "請允許取用相機以進行即時偵測"),
        T("Switch to the back camera", "切换到后置相机", "切換至後置相機"),
        T("Failed to save the captured image to library", "无法保存拍摄的照片到本地", "無法存儲截取的相片到裝置"),
        T("Switch to the front camera", "切换到前置相机", "切換至前置相機"),
        T("{0} could not be loaded. {1}", "{0} 无法加载。{1}", "{0} 無法載入。{1}"),
        T("camera", "相机", "相機"),
        T("loop", "循环", "迴圈"),
        T("frame age", "帧延迟", "影格延遲"),
        T("preprocess", "预处理", "前處理"),
        T("inference", "推理", "推論"),
        T("decode", "解码", "解碼"),
        T("mask", "分割", "遮罩"),
        T("postprocess", "后处理", "後處理"),
        T("end to end", "端到端", "端對端"),
        T("dets", "目标", "物件"),
        T("Style", "样式", "樣式"),
        T("Solid", "纯色", "純色"),
        T("Neon", "霓虹", "霓虹"),
        T("Chip", "标签", "標籤"),
        T("Minimal", "极简", "極簡"),
        T("Overlay", "图层", "圖層"),
        T("Pose overlay", "姿势图层", "姿態圖層"),
        T("Masks", "分割", "遮罩"),
        T("Boxes", "检测框", "偵測框"),
        T("Both", "全部", "全部"),
        T("Skeleton", "骨架", "骨架"),
        T("Body", "身体", "身體"),
        T("Feet", "脚部", "足部"),
        T("Face", "面部", "臉部"),
        T("Hands", "手部", "手部"),
        T("Hand skeleton", "手部骨架", "手部骨架"),
        T("Show stats HUD", "实时状态面板", "即時統計面板"),
        T("box", "检测框", "偵測框"),
        T("size", "大小", "大小"),
    ]

    // MARK: Photo

    private static let photo: [Translation] = [
        T("No photos", "没有照片", "無相片"),
        T("Pick up to 100 images", "最多可选择并导入 100 张图片", "最多可選取並載入 100 張圖片"),
        T("Loading / iCloud", "加载中 / 等待 iCloud", "載入中 / 等待 iCloud"),
        T("Inference", "推理", "推論"),
        T("Select photos", "选择照片", "選取相片"),
        T("{0} selected", "已选择 {0} 张", "已選取 {0} 張"),
        T("Deselect", "取消全选", "取消全選"),
        T("Remove", "移除", "移除"),
        T("Save 1 image to Photos?", "将 1 张图片保存到“照片”？", "將 1 張相片儲存到「照片」？"),
        T("Save {0} images to Photos?", "将 {0} 张图片保存到“照片”？", "將 {0} 張相片儲存到「照片」？"),
        T("Remove 1 photo from this run?", "从当前批次中移除 1 张照片？", "從目前批次中移除 1 張相片？"),
        T("Remove {0} photos from this run?", "从当前批次中移除 {0} 张照片？", "從目前批次中移除 {0} 張相片？"),
        T("The annotated images are added to your photo library.", "带标注的图片将添加到你的照片图库。", "加上標註的圖片將加入你的照片圖庫。"),
        T("The photos stay in your photo library. To have one back in the run, add it again.", "这些照片仍保留在你的照片图库中。如需让某张照片回到批次中，请重新添加。", "這些相片仍保留在你的照片圖庫中。如需讓某張照片回到批次中，請重新載入。"),
        T("Saved 1 image to Photos", "已将 1 张图片保存到“照片”", "已將 1 張相片儲存到「照片」"),
        T("Saved {0} images to Photos", "已将 {0} 张图片保存到“照片”", "已將 {0} 張相片儲存到「照片」"),
        T("Could not save to Photos", "无法保存到“照片”", "無法儲存到「照片」"),
        T("This app is not allowed to add to your photo library, so nothing was saved. Turn on Photos access for it in Settings, then save again.", "本应用没有权限保存照片。请在“设置”中为本应用开启“照片”访问权限，然后重试。", "本 App 未獲准存儲相片。請在「設定」中為本 App 開啟「照片」取用權限，然後重新儲存。"),
        T("Adding to the photo library is restricted on this {0}, for example by Screen Time or a device management profile. Nothing was saved.", "此 {0} 的系统阻止了照片保存，例如受“屏幕使用时间”或设备管理描述文件的限制。保存失败。", "此 {0} 的系統阻止了相片的存儲，例如受「螢幕使用時間」或裝置管理描述檔的限制。儲存失敗。"),
        T("There is not enough free storage on this {0} to save the images.", "此 {0} 的可用存储空间不足，无法保存图片。", "此 {0} 的可用儲存空間不足，無法儲存相片。"),
        T("The photo library reported: {0} ({1} {2})", "照片图库报告：{0}（{1} {2}）", "照片圖庫回報：{0}（{1} {2}）"),
        T("The following image failed to import because it is over {0}:", "以下图片因超过 {0} 而导入失败：", "以下相片因超過 {0} 而載入失敗："),
        T("The following images failed to import because they are over {0}:", "以下图片因超过 {0} 而导入失败：", "以下相片因超過 {0} 而載入失敗："),
        T("and {0} more", "以及另外 {0} 张", "以及另外 {0} 張"),
        T("1 image could not be read.", "有 1 张图片无法读取。", "有 1 張相片無法讀取。"),
        T("{0} images could not be read.", "有 {0} 张图片无法读取。", "有 {0} 張相片無法讀取。"),
        T("Image {0} of {1}", "第 {0} 张（共 {1} 张）", "第 {0} 張（共 {1} 張）"),
        T("ERROR: {0}", "错误：{0}", "錯誤：{0}"),
        T("image", "图片", "相片"),
        T("images", "图片数", "相片數"),
        T("size", "尺寸", "尺寸", context: "image"),
        T("file", "文件", "檔案"),
        T("type", "类型", "類型"),
        T("wall", "总耗时", "總耗時"),
        T("total dets", "目标总数", "物件總數"),
        T("avg e2e", "平均端到端", "平均端對端"),
    ]

    // MARK: Bench and its history

    private static let bench: [Translation] = [
        T("Cold run", "快速测试", "快速測試"),
        T("Sustained", "压力测试", "持續測試"),
        T("Mode", "模式", "模式"),
        T("Unit", "计算单元", "運算單元"),
        T("No benchmarks yet", "暂无基准测试结果", "尚無基準測試結果"),
        T("Run a cold pass across the selected models and compute units.", "对所选的模型和计算单元运行一次快速测试。", "對所選的模型和運算單元執行一次快速測試。"),
        T("All {0} models", "全部 {0} 个模型", "全部 {0} 個模型"),
        T("{0} of {1} models", "已选 {0}/{1} 个模型", "已選 {0}/{1} 個模型"),
        T("All {0} units", "全部 {0} 个单元", "全部 {0} 個單元"),
        T("No units", "未选单元", "未選取單元"),
        T("Deselect all", "取消全选", "取消全選"),
        T("on", "运行于", "執行於"),
        T("Duration: {0} min", "时长：{0} 分钟", "時長：{0} 分鐘"),
        T("Timed iters: {0}", "计时迭代次数：{0}", "計時迭代次數：{0}"),
        T("Warmup: {0}", "预热迭代次数：{0}", "預熱迭代次數：{0}"),
        T("In Sustained mode these set the cold baseline the throttle is measured against.", "在压力测试模式下，这两项用于确定模型的基线，降频幅度将以此为参照。", "在持續測試模式下，這兩項用於確定模型的基準，降頻幅度以此為參照。"),
        T("Per model × unit cell in the cold run.", "快速测试中，每个“模型 × 单元”组合各自执行的次数。", "快速測試中，每個「模型 × 單元」組合各自執行的次數。"),
        T("Paused", "已暂停", "已暫停"),
        T("cold {0}", "冷态 {0}", "冷態 {0}"),
        T("→ live {0} ms", "→ 实时 {0} ms", "→ 即時 {0} ms"),
        T("→ final {0} ms", "→ 最终 {0} ms", "→ 最終 {0} ms"),
        T("Loading {0} @ {1}", "正在加载 {0} @ {1}", "正在載入 {0} @ {1}"),
        T("Benchmarking {0} @ {1}", "正在测试 {0} @ {1}", "正在測試 {0} @ {1}"),
        T("Sustained {0} @ {1}", "持续测试 {0} @ {1}", "持續測試 {0} @ {1}"),
        T("FASTEST", "最快", "最快"),
        T("fastest {0}", "最快：{0}", "最快：{0}"),
        T("sustained", "持续", "持續"),
        T("Cold run · {0} models", "快速测试 · {0} 个模型", "快速測試 · {0} 個模型"),
        T("benchmark.csv ready", "benchmark.csv 已就绪", "benchmark.csv 已就緒"),
        T("History", "历史记录", "歷史紀錄"),
        T("No saved runs", "没有已保存的记录", "沒有已儲存的紀錄"),
        T("Completed cold and sustained runs are saved here.", "完成的快速测试和压力测试会保存在这里。", "完成的快速測試和持續測試會儲存在這裡。"),
        T("Search runs or models", "搜索记录或模型", "搜尋紀錄或模型"),
        T("Sort", "排序", "排序"),
        T("Time", "时间", "時間"),
        T("Name", "名称", "名稱"),
        T("Rename run", "重命名记录", "重新命名紀錄"),
        T("Rename", "重命名", "重新命名"),
        T("Export", "导出", "匯出"),
        T("Delete", "删除", "刪除"),
        T("fastest {0} @ {1} · {2} ms · {3} FPS", "最快：{0} @ {1} · {2} ms · {3} FPS", "最快：{0} @ {1} · {2} ms · {3} FPS"),
        T("throttle +{0}%", "降频 +{0}%", "降頻 +{0}%"),
        T("{0} run exported", "已导出 {0} 条记录", "已匯出 {0} 個紀錄"),
        T("{0} runs exported", "已导出 {0} 条记录", "已匯出 {0} 個紀錄"),
    ]
}
