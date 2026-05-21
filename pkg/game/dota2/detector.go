// Package dota2 implements the zen-lights game.Detector for Dota 2.
//
// Kill score location: top-center HUD, format "RADIANT - DIRE"  e.g. "18 - 5".
// The package registers itself via init(); import with _ to activate:
//
//	import _ "github.com/zen-lights/zen-lights/pkg/game/dota2"
package dota2

import (
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/zen-lights/zen-lights/pkg/game"
)

// scoreRE handles Dota 2's two-number score, separated by a dash.
// Dota 2 uses stylised fonts; OCR can produce en/em dashes.
var scoreRE = regexp.MustCompile(`(\d{1,3})\s*[-–—]\s*(\d{1,3})`)

type detector struct{}

// New returns the Dota 2 Detector.
func New() game.Detector { return &detector{} }

func (d *detector) Name() string { return "dota2" }

// ScoreROI for Dota 2.
//
// The score is at top-center, but slightly wider than LoL due to the font and
// spacing. Tested at 1920×1080 and 2560×1440.
//
//	x: 40%  y: 0.1%  w: 20%  h: 6%
//
// If you are using a custom HUD skin, the position may differ. Run with
// -dump-frames to verify alignment.
func (d *detector) ScoreROI() game.ROI {
	return game.ROI{X: 0.40, Y: 0.001, W: 0.20, H: 0.060}
}

// ParseScore parses "18 - 5" style OCR output into a Score.
// Radiant (green team) kills map to Blue; Dire (red team) kills map to Red.
func (d *detector) ParseScore(ocrText string) (*game.Score, error) {
	text := strings.TrimSpace(ocrText)
	m := scoreRE.FindStringSubmatch(text)
	if m == nil {
		return nil, nil
	}
	radiant, _ := strconv.Atoi(m[1])
	dire, _ := strconv.Atoi(m[2])
	// Dota 2 games rarely exceed 80 kills per side in normal play.
	if radiant > 120 || dire > 120 {
		return nil, nil
	}
	return &game.Score{Blue: radiant, Red: dire}, nil
}

// OCRConfig returns Dota 2-specific OCR settings.
// Dota 2's score uses slightly smaller text with a darker HUD background —
// a threshold of 100 (slightly lower than LoL's 120) captures faint digits
// better. We also use 4× scaling because the score text can be under 20px tall
// on 1080p in some HUD configurations.
func (d *detector) OCRConfig() game.OCRConfig {
	return game.OCRConfig{
		ScaleFactor:    4,
		ThresholdValue: 100,
	}
}

func (d *detector) SampleRate() float64        { return 2.0 }
func (d *detector) MergeWindow() time.Duration { return 10 * time.Second }
func (d *detector) PreBuffer() time.Duration   { return 12 * time.Second }
func (d *detector) PostBuffer() time.Duration  { return 18 * time.Second }

func init() { game.Register(New()) }
