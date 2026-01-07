# Android App (SportsBettingApp)

## Base URL configuration
The Android app uses product flavors to set the FastAPI base URL.

- **Dev flavor**: `BuildConfig.BASE_URL = http://10.0.2.2:8000`
- **Prod flavor**: `BuildConfig.BASE_URL = https://api.example.com`

> **Emulator note:** `10.0.2.2` maps to your host machine’s localhost when using the Android emulator.

Update the values in `mobile/android/SportsBettingApp/app/build.gradle.kts` if you need different endpoints.

## Build flavors
From `mobile/android/SportsBettingApp/`:

```bash
./gradlew assembleDevDebug
./gradlew assembleProdRelease
```

## Required permissions
The app needs network access:

- `android.permission.INTERNET` (declared in `app/src/main/AndroidManifest.xml`)
