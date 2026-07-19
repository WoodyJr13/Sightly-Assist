# Safety, Ethics, and Eligibility Plan

## Prototype status

Sightly Assist is an experimental research prototype. It is not a certified medical device, mobility aid, navigation system, or collision-prevention guarantee.

## Allowed user-facing outputs

The system may output only:

- no alert;
- hazard awareness by broad direction;
- slow;
- stop;
- unable to assess or abstain.

It must not instruct a user to step left, step right, cross a road, enter a space, or treat an unobserved region as safe.

## Fail-safe principles

- Unknown depth is not free space.
- Missing motion is not zero motion.
- Low confidence is surfaced rather than hidden.
- Severe risk may escalate immediately.
- De-escalation is delayed to prevent warning chatter.
- The semantic model cannot override a geometric STOP condition.
- The system records the reason for warnings and abstentions.
- Tests use conservative thresholds until calibration data justify changes.

## Human-participant boundary

The Davidson Technology checklist states that projects involving live vertebrates are not eligible. The exact application of that rule to engineering demonstrations or human-participant evaluation should be confirmed in writing with the program before formal testing is included in the submission.

Until eligibility and any institutional requirements are clarified:

- do not ask blind or low-vision participants to rely on the prototype for mobility;
- do not create collision-risk situations involving people;
- do not claim human-subject performance or usability results;
- use simulation, prerecorded public data, mannequins, nonliving moving rigs, marked trajectories, and supervised non-interventional demonstrations;
- treat any ordinary pedestrian video collection as data collection requiring consent, privacy protection, and eligibility review.

## Controlled-testing rules

- Keep all physical tests below normal walking speed unless a written protocol justifies otherwise.
- Use spotters and barriers where hardware is moving.
- Never test near roads, stairs, platforms, open water, machinery, or uncontrolled traffic.
- Do not obstruct a participant's hearing with continuous audio.
- Do not conceal known failure modes from observers or evaluators.
- Stop a trial immediately after sensor loss, excessive latency, thermal throttling, or unstable output.

## Data privacy

- Prefer local inference and storage.
- Minimize recording of faces and bystanders.
- Obtain permission for staged recordings.
- Separate identifying information from experiment files.
- Document retention and deletion rules.
- Do not upload private recordings to third-party services without explicit permission.
- Use public datasets only under their licenses and intended-use terms.

## Bias and coverage

Evaluation should examine performance variation caused by:

- lighting;
- clothing and contrast;
- object scale and partial occlusion;
- mobility devices;
- crowded scenes;
- camera height and angle;
- indoor versus outdoor conditions;
- detector training-set limitations.

The report should not generalize beyond the tested conditions.

## Audio safety

- Limit digital amplitude and provide a user-controlled output level.
- Use brief cues rather than continuous masking audio.
- Preserve environmental sound.
- Measure cue latency and repetition rate.
- Test exported cues before enabling live playback.
- Document that hearing ability and headphone type affect directional perception.

## Hardware safety

- Secure cables and battery packs.
- Monitor temperature and power draw.
- Use protected batteries and appropriate charging hardware.
- Design breakaway or low-force mounting where worn equipment could snag.
- Record device mass, mounting position, and field of view.

## Claim restrictions

Do not claim that the system:

- prevents collisions;
- identifies a safe route;
- replaces established mobility tools;
- works for all users or environments;
- outputs calibrated probabilities unless calibration is completed;
- has been clinically validated;
- is ready for unsupervised use.

## Required pre-submission review

Before submission, complete:

1. Davidson eligibility clarification for planned evaluation methods.
2. Nominator review of safety and independence.
3. Code and data privacy audit.
4. Model and dataset license audit.
5. Failure-case review.
6. Written limitations section matching the tested evidence.
7. Confirmation that demonstrations do not imply unsupported real-world safety.
