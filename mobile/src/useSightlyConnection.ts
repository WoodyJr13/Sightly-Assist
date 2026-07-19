import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { createLocalDemoState } from './demo';
import type {
  ConnectionMode,
  ConnectionStatus,
  MobileSystemState,
} from './types';

const defaultApiUrl =
  process.env.EXPO_PUBLIC_SIGHTLY_API_URL ?? 'http://192.168.1.100:8765';

export interface SightlyConnection {
  mode: ConnectionMode;
  setMode: (mode: ConnectionMode) => void;
  apiUrl: string;
  setApiUrl: (url: string) => void;
  status: ConnectionStatus;
  state: MobileSystemState;
  history: MobileSystemState[];
  lastError: string | null;
  reconnect: () => void;
}

export function useSightlyConnection(): SightlyConnection {
  const [mode, setMode] = useState<ConnectionMode>('local-demo');
  const [apiUrl, setApiUrl] = useState(defaultApiUrl);
  const [status, setStatus] = useState<ConnectionStatus>('demo');
  const [state, setState] = useState<MobileSystemState>(() => createLocalDemoState(0, 0));
  const [history, setHistory] = useState<MobileSystemState[]>([]);
  const [lastError, setLastError] = useState<string | null>(null);
  const [reconnectToken, setReconnectToken] = useState(0);
  const demoStartedAt = useRef(Date.now());
  const demoFrame = useRef(0);

  const publish = useCallback((next: MobileSystemState) => {
    setState(next);
    setHistory((previous) => [...previous.slice(-89), next]);
  }, []);

  useEffect(() => {
    if (mode !== 'local-demo') {
      return undefined;
    }

    setStatus('demo');
    setLastError(null);
    demoStartedAt.current = Date.now();
    demoFrame.current = 0;
    const update = () => {
      publish(
        createLocalDemoState(
          Date.now() - demoStartedAt.current,
          demoFrame.current++,
        ),
      );
    };
    update();
    const interval = setInterval(update, 250);
    return () => clearInterval(interval);
  }, [mode, publish]);

  const normalizedApiUrl = useMemo(() => normalizeApiUrl(apiUrl), [apiUrl]);

  useEffect(() => {
    if (mode !== 'backend') {
      return undefined;
    }

    setStatus('connecting');
    setLastError(null);
    let active = true;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const socket = new WebSocket(toWebSocketUrl(normalizedApiUrl));

    socket.onopen = () => {
      if (!active) return;
      setStatus('connected');
      setLastError(null);
    };
    socket.onmessage = (event) => {
      if (!active) return;
      try {
        const parsed: unknown = JSON.parse(String(event.data));
        if (!isMobileSystemState(parsed)) {
          throw new Error('Backend sent an incompatible telemetry schema.');
        }
        publish(parsed);
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Invalid backend message.';
        setLastError(message);
        setStatus('error');
      }
    };
    socket.onerror = () => {
      if (!active) return;
      setStatus('error');
      setLastError('Could not reach the Sightly Assist backend.');
    };
    socket.onclose = () => {
      if (!active) return;
      setStatus('disconnected');
      reconnectTimer = setTimeout(() => setReconnectToken((value) => value + 1), 1800);
    };

    return () => {
      active = false;
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      socket.close();
    };
  }, [mode, normalizedApiUrl, publish, reconnectToken]);

  const reconnect = useCallback(() => {
    setReconnectToken((value) => value + 1);
  }, []);

  return {
    mode,
    setMode,
    apiUrl,
    setApiUrl,
    status,
    state,
    history,
    lastError,
    reconnect,
  };
}

function normalizeApiUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  return trimmed || defaultApiUrl;
}

function toWebSocketUrl(apiUrl: string): string {
  if (apiUrl.startsWith('https://')) return `${apiUrl.replace('https://', 'wss://')}/ws/live`;
  if (apiUrl.startsWith('http://')) return `${apiUrl.replace('http://', 'ws://')}/ws/live`;
  return `ws://${apiUrl}/ws/live`;
}

function isMobileSystemState(value: unknown): value is MobileSystemState {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<MobileSystemState>;
  return (
    candidate.schema_version === '1.0' &&
    typeof candidate.frame_id === 'number' &&
    typeof candidate.warning_action === 'string' &&
    typeof candidate.system_message === 'string' &&
    Array.isArray(candidate.hazards) &&
    typeof candidate.metrics === 'object' &&
    candidate.metrics !== null
  );
}
