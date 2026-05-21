// Package pipeline orchestrates the full zen-lights highlight extraction flow:
//
//  1. Probe the input video (ffprobe)
//  2. Stream raw frames at the game's sample rate (FFmpeg → pipe → channel)
//  3. For each frame: crop the score ROI, OCR it, validate and filter the score
//  4. Detect score changes that represent kill events
//  5. Group kill events into highlight Segments
//  6. Cut and losslessly merge those segments into the output file
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

	// DumpFrameDir, when non-empty, causes the preprocessed OCR PNG for every
	// sampled frame to be written to this directory.
	// Useful for tuning ROI coordinates and threshold values.
	// Example: --dump-frames /tmp/zl-debug
	DumpFrameDir string

	// MaxScoreJump is the maximum plausible kill-count increase per sample
	// interval. Readings that exceed this are rejected as OCR noise.
	// Default: 5 (a pentakill in one sample window is the realistic maximum).
	MaxScoreJump int
}

// Result is returned by Run on success.
type Result struct {
	VideoInfo  *ffmpeg.VideoInfo
	Events     []game.KillEvent
	Segments   []highlight.Segment
	OutputPath string
}

// Run executes the highlight extraction pipeline end-to-end.
func Run(cfg Config) (*Result, error) {
	if cfg.MaxScoreJump <= 0 {
		cfg.MaxScoreJump = 5
	}

	// Resolve OCR options: use game-specific overrides if available.
	ocrOpts := ocr.DefaultOptions()
	if provider, ok := cfg.Game.(game.OCRConfigProvider); ok {
		ocrOpts = ocr.OptionsFromConfig(provider.OCRConfig())
	}

	// ── 1. Probe ──────────────────────────────────────────────────────────────
	info, err := ffmpeg.Probe(cfg.InputPath)
	if err != nil {
		return nil, fmt.Errorf("probe: %w", err)
	}
	cfg.logf("video: %dx%d  %.2f fps  duration %s",
		info.Width, info.Height, info.FPS, info.Duration.Round(time.Second))

	d := cfg.Game
	roi := d.ScoreROI().ToImageRect(info.Width, info.Height)
	cfg.logf("game: %s  ROI: %v  sample: %.1f fps", d.Name(), roi, d.SampleRate())

	// ── 2. OCR client ─────────────────────────────────────────────────────────
	ocrClient, err := ocr.New(ocrOpts)
	if err != nil {
		return nil, fmt.Errorf("ocr init: %w", err)
	}
	defer ocrClient.Close()

	// ── 3. Stream frames and detect score changes ─────────────────────────────
	frameCh, errCh := ffmpeg.StreamFrames(info, d.SampleRate())

	var (
		events       []game.KillEvent
		prevScore    *game.Score
		frameCount   int
		noReadStreak int
	)
	dumpPNG := cfg.DumpFrameDir != ""

	for frame := range frameCh {
		frameCount++

		text, debugPNG, err := ocrClient.ReadRegionWithDiagnostics(
			frame.Data, frame.Width, frame.Height, roi, dumpPNG,
		)
		if dumpPNG && debugPNG != nil {
			_ = dumpFramePNG(cfg.DumpFrameDir, frame.Index, frame.Timestamp, debugPNG)
		}
		if err != nil {
			cfg.logf("frame %d OCR error: %v", frame.Index, err)
			continue
		}

		parsed, err := d.ParseScore(text)
		if err != nil {
			cfg.logf("frame %d parse error (text=%q): %v", frame.Index, text, err)
			continue
		}
		if parsed == nil {
			// No score visible (loading screen, replay UI, etc.)
			noReadStreak++
			// After a long gap with no readable score, reset the previous score
			// so we don't incorrectly attribute a score change across a game
			// boundary (e.g. back-to-back VODs or post-game screen).
			if noReadStreak > 30 {
				prevScore = nil
			}
			continue
		}
		noReadStreak = 0

		if prevScore != nil && !prevScore.Equal(*parsed) {
			dBlue := parsed.Blue - prevScore.Blue
			dRed := parsed.Red - prevScore.Red

			// Reject backward or impossibly large jumps — these are OCR noise.
			blueOK := dBlue >= 0 && dBlue <= cfg.MaxScoreJump
			redOK := dRed >= 0 && dRed <= cfg.MaxScoreJump

			if !blueOK || !redOK {
				cfg.logf("frame %d: ignoring implausible score %d-%d → %d-%d",
					frame.Index, prevScore.Blue, prevScore.Red, parsed.Blue, parsed.Red)
				// Don't update prevScore — wait for a plausible reading.
				continue
			}

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

	if streamErr := <-errCh; streamErr != nil {
		return nil, fmt.Errorf("frame stream: %w", streamErr)
	}

	cfg.logf("scanned %d frames, found %d kill event(s)", frameCount, len(events))

	if len(events) == 0 {
		return nil, fmt.Errorf(
			"no kill events detected\n" +
				"  Tip: run with -dump-frames /tmp/zl-debug to inspect preprocessed OCR frames\n" +
				"  Tip: measure the score ROI on a screenshot and update ScoreROI() in the detector\n" +
				"  Tip: if HUD uses dark-on-light text, set OCRConfig().InvertThreshold = true",
		)
	}

	// ── 4. Group events into highlight Segments ───────────────────────────────
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

	// ── 5. Cut and merge ──────────────────────────────────────────────────────
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
