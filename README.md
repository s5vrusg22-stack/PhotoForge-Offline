# PhotoForge Offline

Android photo editor targeting Galaxy S25 Ultra. The application currently implements local photo import, centered transparent-PNG overlay, rotation, horizontal mirror, brightness adjustment and PNG export to Pictures/PhotoForge.

## APK build

1. Open [Actions](https://github.com/s5vrusg22-stack/PhotoForge-Offline/actions).
2. Select **Build Android APK**, then **Run workflow** if the workflow has not already run.
3. After a successful run, download the **PhotoForge-debug-apk** artifact and extract `app-debug.apk`.
4. Install the debug APK on the phone, allowing installation from the selected file manager if prompted.

This workflow uses JDK 17, Android SDK 35, Gradle 8.9 and Android Gradle Plugin 8.7.3.

## Current status and limitations

- Source changes were committed to GitHub, but no successful APK compilation or device test has been verified yet.
- This is a **basic offline photo editor**, not a completed generative-AI app.
- **No LaMa/SDXL/FLUX weights or inference pipeline are integrated into the current GitHub project.**
- High-resolution photos may exhaust Android memory.
- The overlay is centered and scaled automatically; draggable/resizable overlays are future work.
- The Android manifest does not request INTERNET permission.
