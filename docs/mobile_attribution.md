# Mobile Preview Attribution

## Project-specific work

The Sightly Assist mobile milestone adds the following project-specific components:

- `src/sightly_assist/mobile_api.py`: versioned telemetry schemas, replay-result conversion, bounded state publication, LAN HTTP/WebSocket endpoints, and deterministic demonstration telemetry;
- `src/sightly_assist/mobile_cli.py`: local bridge launch and LAN-address reporting;
- `mobile/`: warning-centered interface, free-space and performance displays, object overlays, reconnect behavior, schema checks, speech and haptic controls;
- `scripts/start_expo_preview.ps1` and `scripts/start_expo_preview.sh`: local setup and coordinated preview startup;
- `tests/test_mobile_api.py`: bridge, WebSocket, demo-state, and conversion tests;
- mobile CI: strict TypeScript validation and Android JavaScript bundle export.

## Acquired dependencies

| Dependency | Project role | Attribution status |
|---|---|---|
| FastAPI | Typed HTTP and WebSocket service | Acquired dependency; record version, MIT license, and project URL |
| Uvicorn | Local ASGI server | Acquired dependency; record version, BSD license, and project URL |
| Expo SDK 54 | Mobile runtime and Expo Go tooling | Acquired dependency; record version, MIT license, and project URL |
| React | Component model | Acquired dependency; record version, MIT license, and project URL |
| React Native | Native mobile UI runtime | Acquired dependency; record version, MIT license, and project URL |
| Expo Speech | Spoken warning output | Acquired Expo module; record version, license, and documentation URL |
| Expo Haptics | Haptic warning output | Acquired Expo module; record version, license, and documentation URL |

The project does not claim these frameworks or modules as original work. The original integration contribution is the telemetry contract, conversion from the existing perception results, conservative reconnect and validation behavior, deterministic test scenario, warning-centered information architecture, launch automation, and automated contract testing.

## Demonstration provenance

For each formal demonstration, record:

- repository commit SHA;
- Expo SDK and Expo Go versions;
- phone model and operating-system version;
- whether the data source was phone demo, backend demo, recorded replay, or live hardware;
- computer and phone network configuration;
- speech and haptic settings;
- telemetry update rate and disconnect behavior;
- whether displayed values were synthetic, recorded, or live;
- screenshots or screen recording tied to the tested commit.

## AI assistance

The final Davidson documentation must state what parts were proposed or drafted with AI assistance, what Ved personally reviewed or changed, how behavior was independently tested, and whether Ved can explain and reproduce the mobile bridge and interface architecture.
