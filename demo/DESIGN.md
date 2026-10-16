---
version: alpha
name: DermSSL phone research app
description: Mobile-first research interface for explicitly identified A71 and B71 artifacts.
colors:
  ink: "#202323"
  ink-muted: "#626b6a"
  paper: "#f1f2f2"
  surface: "#ffffff"
  line: "#e3e7e6"
  mint: "#d8eee7"
  mint-strong: "#a8d9c9"
  lavender: "#e8e4f2"
  peach: "#f5e2d8"
  butter: "#f3edce"
  accent: "#326a59"
  accent-dark: "#21493d"
  red: "#933d36"
typography:
  mono:
    fontFamily: '"DM Mono", monospace'
  sans:
    fontFamily: '"DM Sans", sans-serif'
  display:
    fontFamily: '"Newsreader", Georgia, serif'
---

## Scope

The new application lives in `demo/phone`. The older `demo/frontend` is preserved separately and is not the public product interface.

## Components

Use a soft gray canvas, large rounded white cards, pastel mint/lavender/peach accents, and a charcoal prediction panel. Keep evidence charts on light surfaces. Controls have visible keyboard focus and touch targets of at least 44 pixels.

## Evidence and safety

Always distinguish model identity, numerical softmax scores, historical evaluation and current runtime availability. Scores are not clinical confidence. Never enable an unavailable model or claim that unverified segmentation is working. Keep the auxiliary-tool warning visible.

## Motion and offline behavior

Animate transform and opacity only, with short easing and a reduced-motion override. Live capture is sequential and rate bounded, not video streaming. Cache the static shell only, never uploads or API responses. Offline installation does not provide offline inference.
