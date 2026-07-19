# Mathematical Model

## Coordinate convention

The initial simulator uses a two-dimensional ground plane:

- `x` is lateral position, positive to the observer's right;
- `z` is forward position, positive in the observer's forward direction;
- distance is measured in meters;
- time is measured in seconds;
- velocity is measured in meters per second.

## Constant-velocity state

For a body with initial position vector \(\mathbf{p}_0\) and velocity \(\mathbf{v}\), its position at time \(t\) is

\[
\mathbf{p}(t) = \mathbf{p}_0 + \mathbf{v}t.
\]

For an obstacle and observer, define relative position and velocity as

\[
\mathbf{p}_r = \mathbf{p}_{obstacle} - \mathbf{p}_{observer}
\]

and

\[
\mathbf{v}_r = \mathbf{v}_{obstacle} - \mathbf{v}_{observer}.
\]

## Closest point of approach

The unconstrained time of closest approach under constant relative velocity is

\[
t_{CPA} = -\frac{\mathbf{p}_r \cdot \mathbf{v}_r}{\|\mathbf{v}_r\|^2}.
\]

The implementation clips this value to the interval from zero to the configured prediction horizon \(H\):

\[
t_{CPA}^{*} = \operatorname{clip}(t_{CPA}, 0, H).
\]

The predicted separation at that time is

\[
d_{CPA} = \|\mathbf{p}_r + \mathbf{v}_r t_{CPA}^{*}\|.
\]

When relative speed is approximately zero, closest approach is treated as the current time.

## Closing speed

Radial closing speed is

\[
v_c = -\frac{\mathbf{p}_r \cdot \mathbf{v}_r}{\|\mathbf{p}_r\|}.
\]

A positive value means separation is currently decreasing. A negative value means the object is receding.

## Collision boundary

The initial model represents the observer and obstacle as circles. The collision boundary is

\[
r_b = r_{observer} + r_{obstacle} + m,
\]

where \(m\) is a configurable safety margin.

A future boundary crossing is predicted when

\[
d_{CPA} \le r_b
\]

and the closest-approach time is in the future.

## Baseline risk score

The current score combines:

- normalized predicted clearance;
- temporal urgency;
- positive closing speed.

It is deliberately interpretable and bounded between zero and one. It is not a calibrated collision probability. Later milestones will propagate measurement uncertainty and estimate collision probability through repeated trajectory samples.

## Assumptions and limitations

- Motion is constant velocity during the short prediction horizon.
- Bodies are circles on a flat ground plane.
- Position and velocity are exact in simulation.
- Acceleration, curved paths, occlusion, sensor noise, and tracking uncertainty are not yet modeled.
- The current risk level thresholds are provisional engineering values, not validated safety thresholds.
