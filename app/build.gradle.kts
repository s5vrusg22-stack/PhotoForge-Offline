plugins {
 id("com.android.application")
 id("org.jetbrains.kotlin.android")
}
android {
 namespace = "com.example.photoforge"
 compileSdk = 35
 defaultConfig {
  applicationId = "com.example.photoforge"
  minSdk = 29
  targetSdk = 35
  versionCode = 12
  versionName = "1.2"
 }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}
