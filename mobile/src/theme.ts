import type { AlertAction, CorridorStatus } from './types';

export const palette = {
  background: '#07101d',
  surface: '#0d1a2a',
  surfaceRaised: '#13243a',
  border: '#233a55',
  text: '#f5f8fc',
  muted: '#8fa3bb',
  cyan: '#4dd9ff',
  green: '#3ee38f',
  amber: '#ffbd59',
  red: '#ff5c72',
  purple: '#aa8cff',
} as const;

export const warningStyle: Record<
  AlertAction,
  { label: string; color: string; background: string }
> = {
  no_alert: { label: 'PATH CLEAR', color: palette.green, background: '#0d3027' },
  awareness_left: { label: 'HAZARD LEFT', color: palette.cyan, background: '#123044' },
  awareness_center: { label: 'HAZARD AHEAD', color: palette.amber, background: '#3a2b13' },
  awareness_right: { label: 'HAZARD RIGHT', color: palette.cyan, background: '#123044' },
  slow: { label: 'SLOW DOWN', color: palette.amber, background: '#442d0d' },
  stop: { label: 'STOP', color: '#ffffff', background: '#9c1c35' },
  abstain: { label: 'UNABLE TO ASSESS', color: palette.purple, background: '#2e234b' },
};

export function corridorColor(status: CorridorStatus): string {
  if (status === 'clear') return palette.green;
  if (status === 'constrained') return palette.amber;
  if (status === 'blocked') return palette.red;
  return palette.purple;
}

export function formatNumber(value: number | null, unit: string): string {
  return value === null ? '—' : `${value.toFixed(2)} ${unit}`;
}
