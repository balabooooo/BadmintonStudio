# Changelog

All notable changes to Badminton Studio are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-10-04

### Added

- Adjustable voice-command bonus: a slider in the rally panel recomputes scores from a new per-phrase
  point value (0–30) without re-running speech recognition or resegmentation. The rescore API now
  carries the stored bonus and phrase hits, so switching weighting presets no longer drops the bonus.
- Split a clip at a typed time point (`12.5` or `1:23`, with decimals) from the inspector; the unit
  follows the current preview mode (source time / film time), and unparsable input shows a warning.

### Fixed

- Film-preview playback desync after timeline edits. Trimming now ripples following clips, the track
  is magnetically compacted on drag release and project load, the playhead snaps out of gaps,
  cross-media clips switch the video source automatically, clip speed is applied during preview, and
  playback no longer gets stuck at clip boundaries when `timeupdate` samples past a clip end.
- The progress bar now distinguishes film preview (blue, with clip-coverage blocks) from source
  preview (green), with matching mode labels and an on-video badge.

### Changed

- Switching the score scope back to per-media scoring now shows a notification, so score changes are
  never silent.

## [0.1.0] - 2026-09-27

Initial public release.

### Added

- Automatic rally detection and segmentation (multimodal: player movement + frame motion + hit sounds
  + shuttle trajectory), with instant re-segmentation that reuses cached signals.
- Player detection/tracking (YOLO) and pose-assisted cross-court hit attribution to reject neighboring
  court sounds.
- Six-dimension rally scoring (length / intensity / technique / excitement / highlight / picture quality)
  with scoring presets and automatic tags.
- Scene presets (segmentation params + court calibration + preview frame) for reusing tuning per venue.
- Multi-track timeline editor: drag, trim, split, speed change, snapping, undo/redo.
- Export presets (landscape / portrait / square) with portrait auto-follow and NVENC hardware encoding.
- Bilingual UI (Chinese default, English mode), persisted language preference.
- Portable Windows build (PyInstaller, CPU-only PyTorch) for one-click install.

### Changed

- Optimized rally segmentation precision against manual annotations (clip2 +56% F1, clip5 +36% F1).
- Refined rally scoring strategy with a new `highlight_pro` preset and `smash` / `confrontation` /
  `highlight` tags.
