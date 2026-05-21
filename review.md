# zen-lights — Code Review & Improvements

## Problems Found

### 1. OCR preprocessing is too rigid
**File:** `internal/ocr/ocr.go`

The original code hardcodes a single global threshold of `100` and always assumes bright text on a dark background (`Threshold`, never `InvertThreshold`). There was no way to configure this per game, and no adaptive fallback.

**Fix:** Added `Options` / `OCRConfig` structs that each game detector can override. Games now implement the optional `OCRConfigProvider` interface to supply their own scale factor, threshold value, and direction. The `AdaptiveThreshold` (Sauvola-style local thresholding) is available for difficult HUDs with uneven backgrounds.

---

### 2. Score regression and OCR noise were not rejected
**File:** `pkg/pipeline/pipeline.go`

The original pipeline accepted any score change without sanity checks. An OCR misread like `"0 - 0"` mid-game would record a phantom "score reset" event and corrupt everything downstream. Scores could also go backward (OCR confusion), generating invalid `KillEvent` deltas.

**Fix:** Added two validation gates:
- **Backward rejection:** if `parsed.Blue < prevScore.Blue` or `parsed.Red < prevScore.Red`, the frame is skipped and `prevScore` is not updated.
- **Max-jump rejection:** configurable `MaxScoreJump` (default 5) rejects implausibly large single-frame increases (e.g. OCR reading timer digits as "42").

---

### 3. No-read streak didn't reset `prevScore`
**File:** `pkg/pipeline/pipeline.go`

If OCR fails for many consecutive frames (e.g. champion select → cut to gameplay), `prevScore` would be stale from the previous readable frame. The next valid read would diff against a score from minutes earlier and generate fake kill events.

**Fix:** After 30 consecutive frames with no parseable score, `prevScore` is reset to `nil`. This handles VOD boundaries and long loading screens safely.

---

### 4. No debug tooling for ROI tuning
The biggest practical blocker for getting OCR working on a new game or resolution is figuring out the correct `ScoreROI()` values. There was no way to see what the OCR was actually receiving.

**Fix:** Added `-dump-frames <dir>` flag. When set, every preprocessed OCR PNG is written to the directory with filename `frame_000042_0021000ms.png`. You can then open them in any image viewer to verify the ROI crop, threshold direction, and scaling are all correct before running the full pipeline.

---

### 5. ROI coordinates were slightly too narrow
**Files:** `pkg/game/lol/detector.go`, `pkg/game/dota2/detector.go`

The LoL ROI (`W: 0.14`) was fine for the standard client but could clip the score in spectator/broadcast mode, which uses a slightly wider HUD element. The Dota 2 ROI was similarly tight.

**Fix:** Widened both ROIs conservatively. The extra pixels are neutral background after thresholding — Tesseract ignores them — but they prevent clipping issues.

---

### 6. No CS2 / CSGO detector
Adding CS2 demonstrates how to handle a colon-separated score (`8 : 7`) and documents the ROI measurement process in comments.

**Fix:** Added `pkg/game/cs2/detector.go`.

---

### 7. Minor: `CropRGB24` recalculated row offset inside the inner loop
**File:** `internal/imgutil/imgutil.go`

`(y*frameW + x) * 3` was recomputed per pixel. The row base offset `y*frameW*3` was hoisted out of the inner loop.

---

## Files Changed

| File | Change |
|------|--------|
| `internal/imgutil/imgutil.go` | Added `GrayImage` type alias; merged `Threshold`/`InvertThreshold` into shared helper; added `AdaptiveThreshold`; minor crop loop optimisation |
| `internal/ocr/ocr.go` | Added `Options` type built from `game.OCRConfig`; added `ReadRegionWithDiagnostics` for dump-frames support; configurable scale/threshold/direction |
| `pkg/game/game.go` | Added `OCRConfig` struct; added `OCRConfigProvider` optional interface |
| `pkg/game/lol/detector.go` | Wider ROI; implements `OCRConfigProvider` (threshold 120); improved regex (`\d{1,3}`) |
| `pkg/game/dota2/detector.go` | Wider ROI; implements `OCRConfigProvider` (4× scale, threshold 100) |
| `pkg/game/cs2/detector.go` | **New** — colon-separator score, round-count sanity (max 36) |
| `pkg/pipeline/pipeline.go` | Score backward/jump rejection; no-read streak reset; dump-frames integration; actionable error messages |
| `pkg/pipeline/dump.go` | **New** — debug PNG writer |
| `main.go` | Added `-dump-frames` and `-max-score-jump` flags; registered CS2 |

---

## Usage After Changes

```bash
# Basic run (unchanged)
./zenlights -game lol -input match.mp4 -output highlights.mp4 -v

# Debug OCR — inspect preprocessed frames to tune ROI
./zenlights -game lol -input match.mp4 -output /dev/null -dump-frames /tmp/zl-debug
# Then open /tmp/zl-debug/frame_*.png — you should see clean black-and-white digits

# If highlights are too noisy (OCR artifacts), tighten the jump guard
./zenlights -game lol -input match.mp4 -output out.mp4 -max-score-jump 3
```

## Next Steps (not in this PR)

- **Template matching as OCR fallback** — for games where Tesseract is unreliable, use pixel-pattern matching against known digit sprites extracted from screenshots. Far more accurate for fixed HUDs, zero Tesseract dependency.
- **Confidence scoring** — expose Tesseract's per-character confidence and skip frames below a threshold (e.g. < 70%) instead of relying purely on sanity checks.
- **Multi-ROI support** — some games show the score in multiple places (score + timer); reading two ROIs and cross-validating eliminates most remaining false positives.
- **Valorant / Overwatch detectors** — both use bright digits on dark HUDs similar to LoL; the ROI is the main thing to measure.
