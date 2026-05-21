// Package pipeline orchestrates the full zen-lights highlight extraction flow:
//
//  1. Probe the input video (ffprobe)
//  2. Stream raw frames at the game's sample rate (FFmpeg → pipe → channel)
//  3. For each frame: crop the score ROI, OCR it, detect score changes
//  4. Group kill events into highlight Segments
//  5. Cut and losslessly merge those segments into the output file
package pipeline

import (
	"fmt"
	"log"
	"time"

	"github.com/zen-lights/zen-lights/internal/ffmpeg"
	"github.com/zen-lights/zen-lights/internal/ocr"
	"github.com/zen-lights/zen-lights/pkg/game"
	"github.com/zen-lights/zen-lights/pkg/highlight"
)

// Config controls a single pipeline run.
type Config struct {
	InputPath  string
	OutputPath string
	Game       game.Detector
	Verbose    bool
}

// Result is returned by Run on success.
type Result struct {
	VideoInfo  *ffmpeg.VideoInfo
	Events     []game.KillEvent
	Segments   []highlight.Segment
	OutputPath string
}

// Run executes the highlight extraction pipeline end-to-end.
//
// The only files written to disk are:
//   - The final merged output at cfg.OutputPath
//   - Short-lived temp part files under os.TempDir() (removed automatically)
//
// No frames are ever written to disk.
func Run(cfg Config) (*Result, error) {
	// ── 1. Probe ────────────────────────────────────────────────────────────
	info, err := ffmpeg.Probe(cfg.InputPath)
	if err != nil {
		return nil, fmt.Errorf("probe: %w", err)
	}
	cfg.logf("video: %dx%d  %.2f fps  duration %s",
		info.Width, info.Height, info.FPS, info.Duration.Round(time.Second))

	d := cfg.Game
	roi := d.ScoreROI().ToImageRect(info.Width, info.Height)
	cfg.logf("game: %s  ROI: %v  sample: %.1f fps", d.Name(), roi, d.SampleRate())

	// ── 2. OCR client (one Tesseract session for the whole run) ─────────────
	ocrClient, err := ocr.New()
	if err != nil {
		return nil, fmt.Errorf("ocr init: %w", err)
	}
	defer ocrClient.Close()

	// ── 3. Stream frames and detect score changes ───────────────────────────
	frameCh, errCh := ffmpeg.StreamFrames(info, d.SampleRate())

	var (
		events     []game.KillEvent
		prevScore  *game.Score
		frameCount int
	)

	for frame := range frameCh {
		frameCount++

		text, err := ocrClient.ReadRegion(frame.Data, frame.Width, frame.Height, roi)
		if err != nil {
			cfg.logf("frame %d OCR error: %v", frame.Index, err)
			continue
		}

		parsed, err := d.ParseScore(text)
		if err != nil {
			cfg.logf("frame %d parse error: %v", frame.Index, err)
			continue
		}
		if parsed == nil {
			// No score visible in this frame (loading screen, post-game, etc.)
			continue
		}

		if prevScore != nil && !prevScore.Equal(*parsed) {
			ev := game.KillEvent{
				At:          frame.Timestamp,
				ScoreBefore: *prevScore,
				ScoreAfter:  *parsed,
			}
			events = append(events, ev)
			cfg.logf("kill @ %s  %d-%d → %d-%d  (Δ%d)",
				frame.Timestamp.Round(time.Millisecond),
				ev.ScoreBefore.Blue, ev.ScoreBefore.Red,
				ev.ScoreAfter.Blue, ev.ScoreAfter.Red,
				ev.Delta())
		}
		prevScore = parsed
	}

	// Surface any stream error (e.g. corrupt video, broken pipe)
	if streamErr := <-errCh; streamErr != nil {
		return nil, fmt.Errorf("frame stream: %w", streamErr)
	}

	cfg.logf("scanned %d frames, found %d kill event(s)", frameCount, len(events))

	if len(events) == 0 {
		return nil, fmt.Errorf("no kill events detected — check ROI coordinates or video content")
	}

	// ── 4. Group events into highlight Segments ──────────────────────────────
	segments := highlight.GroupEvents(
		events,
		d.MergeWindow(),
		d.PreBuffer(),
		d.PostBuffer(),
		info.Duration,
	)

	if cfg.Verbose {
		for i, seg := range segments {
			log.Printf("  segment %d: %s", i+1, seg)
		}
	}

	// ── 5. Cut and merge ─────────────────────────────────────────────────────
	ffsegs := make([]ffmpeg.Segment, len(segments))
	for i, s := range segments {
		ffsegs[i] = ffmpeg.Segment{Start: s.Start, End: s.End}
	}

	cfg.logf("cutting %d segment(s) → %s", len(ffsegs), cfg.OutputPath)
	if err := ffmpeg.CutAndMerge(cfg.InputPath, ffsegs, cfg.OutputPath); err != nil {
		return nil, fmt.Errorf("cut & merge: %w", err)
	}

	cfg.logf("done ✓")

	return &Result{
		VideoInfo:  info,
		Events:     events,
		Segments:   segments,
		OutputPath: cfg.OutputPath,
	}, nil
}

func (c Config) logf(format string, args ...any) {
	if c.Verbose {
		log.Printf("[zen-lights] "+format, args...)
	}
}
