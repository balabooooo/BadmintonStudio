# Changelog

All notable changes to Badminton Studio are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning follows
[Semantic Versioning](https://semver.org/).

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
