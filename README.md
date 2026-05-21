# zen-lights

**Single-pass MOBA gameplay highlight extractor written in Go.**  
Detects kill events from the HUD score, cuts the clips, and merges them — no re-encoding, no temp frames on disk.

---

## Features

- 🎮 **Modular game support** — add a new game by implementing one interface
- ⚡ **Zero disk frame writes** — FFmpeg pipes raw frames directly into memory
- ✂️ **Lossless cutting** — stream copy via FFmpeg concat demuxer, near-instant
- 🔍 **Single-pass detection** — one video read for all timestamps, then cut
- 🌐 **HTTP preview** — browser-based player with seekable segment list

## Supported games

| Flag     | Game                  |
|----------|-----------------------|
| `lol`    | League of Legends     |
| `dota2`  | Dota 2                |

---

## Requirements

| Dependency     | Install                                      |
|----------------|----------------------------------------------|
| Go ≥ 1.21      | https://go.dev/dl                            |
| FFmpeg + ffprobe | `brew install ffmpeg` / `apt install ffmpeg` |
| Tesseract OCR  | `brew install tesseract` / `apt install tesseract-ocr` |

---

## Quick start

```bash
# Clone and build
git clone https://github.com/zen-lights/zen-lights
cd zen-lights
go build -o zenlights ./cmd/zenlights

# Extract highlights from a LoL VOD
./zenlights -game lol -input match.mp4 -output highlights.mp4 -v

# With browser preview (opens http://localhost:8765)
./zenlights -game lol -input match.mp4 -output highlights.mp4 -preview
```

---

## Adding a new game

1. Create `pkg/game/<yourgame>/detector.go`
2. Implement `game.Detector` (6 methods — see below)
3. Call `game.Register(New())` in `init()`
4. Blank-import in `cmd/zenlights/main.go`

```go
package mygame

import (
    "regexp"
    "strconv"
    "time"
    "github.com/zen-lights/zen-lights/pkg/game"
)

var scoreRE = regexp.MustCompile(`(\d+)\s*[-–]\s*(\d+)`)

type detector struct{}

func New() game.Detector { return &detector{} }

func (d *detector) Name() string { return "mygame" }

// ScoreROI: measure in a screenshot — express as fractions of frame size
func (d *detector) ScoreROI() game.ROI {
    return game.ROI{X: 0.44, Y: 0.004, W: 0.12, H: 0.04}
}

func (d *detector) ParseScore(text string) (*game.Score, error) {
    m := scoreRE.FindStringSubmatch(text)
    if m == nil { return nil, nil }
    a, _ := strconv.Atoi(m[1])
    b, _ := strconv.Atoi(m[2])
    return &game.Score{Blue: a, Red: b}, nil
}

func (d *detector) SampleRate() float64       { return 2.0 }
func (d *detector) MergeWindow() time.Duration { return 8 * time.Second }
func (d *detector) PreBuffer() time.Duration   { return 10 * time.Second }
func (d *detector) PostBuffer() time.Duration  { return 15 * time.Second }

func init() { game.Register(New()) }
```

---

## Architecture

```
Input video
    │
    ▼
ffmpeg.Probe()          — width, height, fps, duration (ffprobe)
    │
    ▼
ffmpeg.StreamFrames()   — raw RGB24 frames at 2 fps via stdout pipe
    │                     (no temp files, frames live only in memory)
    ▼
ocr.ReadRegion()        — crop ROI → grayscale → 3× scale → threshold → Tesseract
    │
    ▼
game.Detector.ParseScore() — regex parse "12 - 8" → Score
    │
    ▼
score diff detection    — emit KillEvent on score change
    │
    ▼
highlight.GroupEvents() — merge nearby kills into Segments
    │
    ▼
ffmpeg.CutAndMerge()    — stream-copy each segment → concat → output.mp4
                          (only disk write: final output + short-lived part files in /tmp)
```

---

## Flags

| Flag       | Default           | Description                          |
|------------|-------------------|--------------------------------------|
| `-input`   | *(required)*      | Input video path                     |
| `-output`  | `highlights.mp4`  | Output video path                    |
| `-game`    | *(required)*      | Game name (`lol`, `dota2`, …)        |
| `-preview` | false             | Launch HTTP preview after processing |
| `-addr`    | `localhost:8765`  | Preview server address               |
| `-v`       | false             | Verbose logging                      |

---

## Tuning OCR accuracy

If kill detection misses events:

1. **Check the ROI** — screenshot a frame, measure the score position in pixels,
   convert to fractions and update `ScoreROI()` in the game's detector.
2. **Threshold direction** — if the HUD uses dark text on light background,
   swap `imgutil.Threshold` → `imgutil.InvertThreshold` in `ocr/ocr.go`.
3. **Scale factor** — increase from `3` to `4` in `ocr.go` for very small HUD text.
4. **Sample rate** — increase `SampleRate()` from 2 to 4 fps if kills are missed
   (costs proportionally more CPU).

---

## Performance notes

- **CPU**: FFmpeg does all decode/resample work; Go only reads the pipe and crops a tiny ROI per frame. Expect < 20% of one core for a 1080p file at 2 fps sample rate.
- **Memory**: ~3 frame buffers live at a time (ring buffer depth 16 × framesize). For 1080p RGB24 that's 16 × 6 MB = ~96 MB peak.
- **Disk**: Zero writes during detection. Only `/tmp/zenlights-*/partNNNN.mp4` files exist briefly during the final cut+merge step.
- **Speed**: Detection runs faster-than-realtime. Cutting is instant (stream copy).
