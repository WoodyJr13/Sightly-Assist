# Sightly Assist Expo Go Preview

This application is the phone-based monitoring and demonstration interface for the Sightly Assist research backend.

## What works now

- self-contained phone demo with an approaching and crossing-person sequence;
- warning-first dashboard;
- normalized object-box overlays;
- object distance, velocity, risk, CPA time, and CPA distance;
- left, center, and right free-space status;
- pipeline FPS and stage latency;
- spoken and haptic warning tests;
- LAN WebSocket connection to the Python mobile bridge;
- automatic backend reconnection and telemetry schema validation.

The app is a monitoring and research interface. It does not run the ONNX detector inside Expo Go.

## Prerequisites

- Node.js 20 or later;
- Expo Go installed on the phone;
- phone and computer on the same local network;
- Python 3.11 or later for backend mode.

During the current Expo SDK transition, the project intentionally targets Expo SDK 54 because that is the physical-device Expo Go-compatible SDK.

## Phone-only test

```bash
cd mobile
npm install
npm start
```

Scan the QR code with Expo Go. The app begins in **Phone Demo** mode and does not require the Python backend.

## Backend-connected test

From the repository root:

```bash
python -m pip install -e ".[dev,visualization,mobile]"
sightly-mobile serve --demo
```

The command prints one or more LAN URLs such as:

```text
http://192.168.1.42:8765
```

In a second terminal:

```bash
cd mobile
npm install
npm start
```

Open **Settings**, enter the printed LAN URL, select **Backend**, and press **Connect / Retry**.

Do not enter `localhost` in the phone app. On a physical phone, `localhost` refers to the phone itself.

## Environment configuration

A default backend can be supplied before starting Expo:

### PowerShell

```powershell
$env:EXPO_PUBLIC_SIGHTLY_API_URL="http://192.168.1.42:8765"
npm start
```

### macOS/Linux

```bash
EXPO_PUBLIC_SIGHTLY_API_URL=http://192.168.1.42:8765 npm start
```

## Backend endpoints

- `GET /health`
- `GET /api/v1/state`
- `GET /api/v1/history?limit=100`
- `POST /api/v1/demo/reset`
- `WS /ws/live`

## Real pipeline integration

The backend owns a `MobileStateStore`. A live or replay pipeline publishes:

```python
store.publish(
    mobile_state_from_replay_result(
        result,
        source="oakd-live",
        sequence_id="controlled-test-001",
    )
)
```

The app does not duplicate detection, tracking, depth, motion, or collision calculations.

## Current limits

- the camera panel is a telemetry overlay, not a streamed RGB image yet;
- the built-in demo contains synthetic values;
- real hardware inference still requires the Python/OAK-D backend;
- settings are not yet persisted after reloading Expo Go;
- background operation and custom low-latency native audio require a later Expo development build;
- the system is not validated for independent real-world mobility use.
