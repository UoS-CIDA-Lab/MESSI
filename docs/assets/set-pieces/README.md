# Rare set-piece previews

These short clips are synthetic restart diagnostics rendered by the FootballWorld
v1.0.0 reference policy. They are not source-data clips and are not evidence that a
specific restart occurred in the calibration data. Only the initial restart
boundary is staged; every displayed transition after that boundary is produced
by the normal rule policy, physics, and laws engine.

All clips use base seed `907`, an 80 Hz physics clock, 10 Hz policy decisions,
and 20 rendered frames per second. Each MP4 contains 100 decoded H.264/yuv420p
frames at 1280 x 720 (5.0 seconds). Full-file decoding verified frame count,
duration, rate, dimensions, codec, and pixel format. The frozen Python-source
fingerprint for the capture is
`ac7619bdcdc62da4be618bcde7d6206a5004c60c3550fd9685b297c6c0db0f9c`.

- [Throw-in](throw-in.mp4)
- [Goal kick](goal-kick.mp4)
- [Corner kick](corner.mp4)
- [Direct free kick](free-kick.mp4)
- [Penalty kick](penalty.mp4)
- [Offside indirect free kick](offside-free-kick.mp4)
- [Goalkeeper hold and distribution](gk-hold.mp4)

The production three-second restart-release clock is retained. Ordinary first
releases occur on control frames 30--33; the goalkeeper-hold fixture begins
with its release on frame 0. Across all seven clips, physics event-budget
exhaustion and unintended boundary exits are both zero. The corner produces a
reachable aerial service with one `FOOT` release and no same-taker
`PASSIVE_BODY` retouch.

| Clip | SHA-256 |
| --- | --- |
| Throw-in | `ac7825ed853a84320debdf770a06fe232f2148608e1a54657cde33b64e9648ed` |
| Goal kick | `91ed8a0df525489b8434b0358f104ac3d5037103b3fd2117290fb0636dd912c6` |
| Corner kick | `fb7bd7677ca530d542695e5cc48008e8696861c368a1897aed50f802261a1d8a` |
| Direct free kick | `ee4b83c37a007a725bc3031f4c0079b54dbc5298016b5425a2e0e27a73e6098f` |
| Penalty kick | `c72011613220086434bd65577f2f78bee4a8b00911b91360acbc4dbd80b97041` |
| Offside indirect free kick | `5b8ee01797b49582aa0d68a417e8ba0af793fc5ca31260b57fe95a1deee975f7` |
| Goalkeeper hold | `0c79720aa9ce6f0f8111026f9b7b3fb289c295fe2f5781a91690373243faa384` |

The scene generator, raw event/tracking sidecars, and any data-processing or
fitting pipeline are intentionally excluded from the public distribution.

The capture boundary follows the sound SoccerWorld showcase convention:
construct one non-transition restart boundary, then exercise real transitions.
FootballWorld deliberately publishes only the verified videos and this bounded
receipt instead of inheriting the public scene generator and raw sidecars. That
change preserves the private validation and calibration boundary.
