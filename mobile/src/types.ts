export type AlertAction =
  | 'no_alert'
  | 'awareness_left'
  | 'awareness_center'
  | 'awareness_right'
  | 'slow'
  | 'stop'
  | 'abstain';

export type CorridorStatus = 'clear' | 'constrained' | 'blocked' | 'unknown';

export interface MobileBoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

export interface MobileHazard {
  track_id: number;
  class_name: string;
  confidence: number;
  bbox: MobileBoundingBox;
  depth_m: number | null;
  velocity_x_mps: number | null;
  velocity_z_mps: number | null;
  speed_mps: number | null;
  risk_score: number | null;
  time_to_closest_approach_s: number | null;
  distance_at_closest_approach_m: number | null;
  predicted_collision: boolean | null;
  assessment_status: string;
}

export interface MobileFreeSpace {
  left: CorridorStatus;
  center: CorridorStatus;
  right: CorridorStatus;
}

export interface MobileMetrics {
  processing_fps: number;
  total_latency_ms: number;
  detector_latency_ms: number;
  tracker_latency_ms: number;
  depth_latency_ms: number;
  motion_latency_ms: number;
  risk_latency_ms: number;
}

export interface MobileSystemState {
  schema_version: '1.0';
  source: string;
  sequence_id: string;
  frame_id: number;
  timestamp_ns: number;
  warning_action: AlertAction;
  warning_should_emit: boolean;
  warning_reason: string;
  free_space: MobileFreeSpace;
  hazards: MobileHazard[];
  metrics: MobileMetrics;
  synchronized: boolean;
  orientation_available: boolean;
  system_message: string;
}

export type ConnectionMode = 'local-demo' | 'backend';
export type ConnectionStatus = 'demo' | 'connecting' | 'connected' | 'disconnected' | 'error';
