# Risk Register

| Risk | Impact | Current mitigation | Status |
|---|---|---|---|
| Incorrect collision mathematics | Unsafe or misleading conclusions | Analytical scenarios and property tests before sensor integration | Active |
| Risk score mistaken for probability | Invalid scientific claims | Label as an uncalibrated score until calibration experiments exist | Controlled |
| Detector or depth failure | Missed or false hazard alerts | Future sensor-health checks, uncertainty propagation, and abstention | Planned |
| Camera ego-motion mistaken for object motion | Incorrect trajectory estimates | Separate observer and obstacle motion; add IMU/visual odometry later | Planned |
| VLM hallucination | Incorrect semantic context | VLM cannot issue or override safety actions; strict schema and fallback | Planned |
| Unsafe directional command | User could move into another hazard | Directional guidance disabled by default; future corridor hard constraints | Controlled |
| Dataset leakage | Inflated evaluation results | Split by session, environment, and object instance; preserve untouched test set | Planned |
| Human-subject or privacy violation | Ethical and legal harm | No human testing without proper supervision/approval; local-only recording defaults | Planned |
| Thermal throttling on edge hardware | Increased latency and missed warnings | Sustained benchmarks, telemetry, cooling documentation | Planned |
| Unsupported model export | Deployment delay | Runtime abstraction and benchmark multiple model/runtime paths | Planned |
| Overreliance on generated code | Weak understanding or contribution record | Maintain student contribution and decision logs; manually review all components | Active |
