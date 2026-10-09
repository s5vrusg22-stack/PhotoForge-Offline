plugins {
 id("com.android.application")
 id("org.jetbrains.kotlin.android")
}
android {
 namespace = "com.example.photoforge"
 compileSdk = 35
 compileOptions {
  sourceCompatibility = JavaVersion.VERSION_17
  targetCompatibility = JavaVersion.VERSION_17
 }
 kotlinOptions {
  jvmTarget = "17"
 }
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
    implementation("com.google.ai.edge.litert:litert:2.1.0")
    testImplementation("junit:junit:4.13.2")
}
