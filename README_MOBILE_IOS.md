# SportsBetting iOS App

## Base URL configuration
The app reads `BASE_URL` from `mobile/ios/SportsBettingApp/SportsBettingApp/Info.plist`.

- **DEV (simulator):** set `BASE_URL` to `http://localhost:8000` if FastAPI runs on your Mac.
- **DEV (physical iPhone):** use your Mac's LAN IP (e.g. `http://192.168.1.10:8000`).
- **PROD:** set `BASE_URL` to your hosted API domain.

> Tip: to discover your LAN IP, run `ipconfig getifaddr en0` on macOS.

## Running in the simulator
1. Open `mobile/ios/SportsBettingApp/SportsBettingApp.xcodeproj` in Xcode.
2. Select an iOS 16+ simulator.
3. Build and run.

## ATS / CORS notes
- iOS blocks plain HTTP by default. The project includes an `NSAppTransportSecurity` exception for `localhost` in `Info.plist` to enable local testing.
- For LAN IP testing, add an exception entry for your IP/domain or switch to HTTPS.
- If you see CORS errors from the app, configure CORS in your FastAPI server to allow the device's origin.
