// Package lol implements the zen-lights game.Detector for League of Legends.
//
// Kill score location: top-center HUD, format "BLUE - RED"  e.g. "12 - 8".
// The package registers itself via init(); import with _ to activate:
//
//	import _ "github.com/zen-lights/zen-lights/pkg/game/lol"
package lol

import (
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/zen-lights/zen-lights/pkg/game"
)

// scoreRE matches the LoL HUD kill score: two numbers separated by a dash.
// OCR can produce en-dashes or extra whitespace, so we handle those variants.
// Also matches "0 - 0" at game start to establish a baseline.
var scoreRE = regexp.MustCompile(`(\d{1,3})\s*[-–—]\s*(\d{1,3})`)

type detector struct{}

// New returns the LoL Detector.
func New() game.Detector { return &detector{} }

func (d *detector) Name() string { return "lol" }

// ScoreROI for LoL: the kill score sits in the top-center.
//
// Measured at 1920×1080, 2560×1440, and 3840×2160.
// Two ROI variants exist depending on client version / spectator mode:
//   - Normal game client:  x≈43%, y≈0.3%, w≈14%, h≈4.5%
//   - Spectator/broadcast: x≈42%, y≈0.2%, w≈16%, h≈5.5%
//
// The broader ROI here covers both variants; the extra width only adds a thin
// strip of background pixels that Tesseract ignores after thresholding.
func (d *detector) ScoreROI() game.ROI {
	return game.ROI{X: 0.42, Y: 0.002, W: 0.16, H: 0.055}
}

// ParseScore parses "12 - 8" style OCR text into a Score.
// Returns nil, nil if no score pattern is found (loading screen, champion select, etc).
func (d *detector) ParseScore(ocrText string) (*game.Score, error) {
	text := strings.TrimSpace(ocrText)
	m := scoreRE.FindStringSubmatch(text)
	if m == nil {
		return nil, nil
	}
	blue, _ := strconv.Atoi(m[1])
	red, _ := strconv.Atoi(m[2])
	// Sanity check: LoL games don't exceed ~100 kills per side.
	// High values indicate a misread (e.g. timer digits leaked into the ROI).
	if blue > 100 || red > 100 {
		return nil, nil
	}
	return &game.Score{Blue: blue, Red: red}, nil
}

// OCRConfig returns LoL-specific OCR preprocessing settings.
// LoL uses bright white digits on a dark semi-transparent HUD background —
// the default (bright-on-dark, threshold=110, 3× scale) works well.
// We bump the threshold slightly to 120 to cut through the semi-transparent
// background gradient that often bleeds into the ROI.
func (d *detector) OCRConfig() game.OCRConfig {
	return game.OCRConfig{
		ThresholdValue: 120,
	}
}

func (d *detector) SampleRate() float64        { return 2.0 }
func (d *detector) MergeWindow() time.Duration { return 8 * time.Second }
func (d *detector) PreBuffer() time.Duration   { return 10 * time.Second }
func (d *detector) PostBuffer() time.Duration  { return 15 * time.Second }

func init() { game.Register(New()) }
