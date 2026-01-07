# Android App (SportsBettingApp)

## Base URL configuration
The Android app uses product flavors to set the FastAPI base URL.

- **Dev flavor**: `BuildConfig.BASE_URL = http://10.0.2.2:8000`
- **Prod flavor**: `BuildConfig.BASE_URL = https://CHANGE_ME`

> **Emulator note:** `10.0.2.2` maps to your host machine’s localhost when using the Android emulator.

Update the values in `mobile/android/SportsBettingApp/app/build.gradle.kts` if you need different endpoints.

## Cleartext HTTP (dev only)
The dev flavor enables cleartext HTTP via `app/src/dev/AndroidManifest.xml` (`android:usesCleartextTraffic="true"`).
Prod does not allow cleartext by default.

## Build & run
From `mobile/android/SportsBettingApp/`:

```bash
./gradlew assembleDevDebug
```

Or open the project in Android Studio and run the **devDebug** variant on an emulator (API 26+).

## Required permissions
The app needs network access:

- `android.permission.INTERNET` (declared in `app/src/main/AndroidManifest.xml`)

## What the app does
- Run tab: choose sport + date, then POST `/api/runs` and navigate to predictions.
- Predictions: GET `/api/runs/{run_id}/predictions` and provide an “Open CSV” button.
- Bets: filter by date + sport and GET `/api/bets`.
- Settings: base URL, version, and disclaimer.
