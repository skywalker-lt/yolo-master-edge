import java.util.Properties

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
}

// Per-machine SDK roots for the native build (ncnn Android + opencv-mobile).
// Copy android/sdk-paths.example.properties -> android/sdk-paths.properties (gitignored).
val sdkPaths = Properties().apply {
    val f = rootProject.file("sdk-paths.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}
val ncnnRoot: String = sdkPaths.getProperty("NCNN_ANDROID_ROOT", System.getenv("NCNN_ANDROID_ROOT") ?: "")
val opencvRoot: String = sdkPaths.getProperty("OPENCV_ANDROID_ROOT", System.getenv("OPENCV_ANDROID_ROOT") ?: "")

android {
    namespace = "dev.yolomaster.ncnn"
    compileSdk = 34
    ndkVersion = "26.3.11579264"

    defaultConfig {
        minSdk = 24 // libc++fs + Vulkan 1.1
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DNCNN_ANDROID_ROOT=$ncnnRoot",
                    "-DOPENCV_ANDROID_ROOT=$opencvRoot",
                    "-DANDROID_STL=c++_shared",
                )
                cppFlags += "-std=c++17"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    sourceSets["main"].java.srcDir("src/main/kotlin")
    sourceSets["androidTest"].java.srcDir("src/androidTest/kotlin")

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
}
