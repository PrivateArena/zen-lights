// Package cs2 implements the zen-lights game.Detector for Counter-Strike 2 (and CS:GO).
//
// Kill score location: top-center HUD.
// Score format: "CT_SCORE  :  T_SCORE"  e.g. "8  :  7"  (colon separator, not dash).
// Broadcast/HLTV overlays use the same layout.
//
// The package registers itself via init(); import with _ to activate:
//
//	import _ "github.com/zen-lights/zen-lights/pkg/game/cs2"
package cs2

import (
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/zen-lights/zen-lights/pkg/game"
)

// scoreRE matches "8 : 7" or "8:7" — CS2 uses a colon, not a dash.
// Also handles OCR artifacts like "8 - 7" for robustness.
var scoreRE = regexp.MustCompile(`(\d{1,2})\s*[:–-]\s*(\d{1,2})`)

type detector struct{}

// New returns the CS2 Detector.
func New() game.Detector { return &detector{} }

func (d *detector) Name() string { return "cs2" }

// ScoreROI for CS2/CSGO.
//
// The round score is top-center on the default HUD and on most tournament
// overlays (ESL, BLAST, PGL). Tested at 1920×1080 and 2560×1440.
//
//	x: 43%  y: 0.0%  w: 14%  h: 7%
//
// Note: CS2 shows the score at the very top of the frame (y ≈ 0), so
// the ROI starts from row 0.
func (d *detector) ScoreROI() game.ROI {
	return game.ROI{X: 0.43, Y: 0.0, W: 0.14, H: 0.07}
}

// ParseScore parses "8 : 7" style OCR text into a Score.
// CT side maps to Blue; T side maps to Red.
// Returns nil, nil if no recognisable round score is found.
func (d *detector) ParseScore(ocrText string) (*game.Score, error) {
	text := strings.TrimSpace(ocrText)
	m := scoreRE.FindStringSubmatch(text)
	if m == nil {
		return nil, nil
	}
	ct, _ := strconv.Atoi(m[1])
	t, _ := strconv.Atoi(m[2])
	// Max rounds in a CS2 match is 30 (MR15); overtime adds at most 6 more.
	if ct > 36 || t > 36 {
		return nil, nil
	}
	return &game.Score{Blue: ct, Red: t}, nil
}

// OCRConfig for CS2: the score digits are large and white on a dark background.
// 3× scale and threshold 115 work reliably across common resolutions.
func (d *detector) OCRConfig() game.OCRConfig {
	return game.OCRConfig{
		ThresholdValue: 115,
	}
}

// CS2-specific timing: rounds are short, so tighter buffers still feel complete.
func (d *detector) SampleRate() float64        { return 2.0 }
func (d *detector) MergeWindow() time.Duration { return 15 * time.Second } // whole round chain
func (d *detector) PreBuffer() time.Duration   { return 8 * time.Second }
func (d *detector) PostBuffer() time.Duration  { return 12 * time.Second }

func init() { game.Register(New()) }
