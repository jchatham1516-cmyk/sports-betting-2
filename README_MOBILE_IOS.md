# SportsBetting iOS App

## Base URL configuration
The app reads `BASE_URL` from `mobile/ios/SportsBettingApp/SportsBettingApp/Info.plist`.

- **Simulator (default):** `http://127.0.0.1:8000`
- **Physical iPhone:** set `BASE_URL` to your Mac’s LAN IP (for example, `http://192.168.1.10:8000`).
- **Prod:** point `BASE_URL` to your hosted API domain.

> Tip: to discover your LAN IP, run `ipconfig getifaddr en0` on macOS.

## Running in the simulator
1. Open `mobile/ios/SportsBettingApp/SportsBettingApp.xcodeproj` in Xcode.
2. Select an iOS 16+ simulator.
3. Build and run.

## ATS / HTTP notes
- Local development uses plain HTTP. `Info.plist` includes `NSAppTransportSecurity` with `NSAllowsArbitraryLoads = true` for dev/testing.
- If you switch to a physical device, ensure the LAN IP is reachable and update `BASE_URL` accordingly.

## What the app does
- Run tab: choose sport + date, then POST `/api/runs` and navigate to predictions.
- Predictions: GET `/api/runs/{run_id}/predictions` and provide an “Open CSV” button.
- Bets: filter by date + sport and GET `/api/bets`.
- Settings: base URL, version, and disclaimer.
