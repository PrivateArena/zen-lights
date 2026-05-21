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
// OCR can produce en-dashes or extra spaces, so we handle those variants.
var scoreRE = regexp.MustCompile(`(\d+)\s*[-–—]\s*(\d+)`)

type detector struct{}

// New returns the LoL Detector. Prefer relying on the auto-registered instance
// via game.Get("lol"); use New() only when you need a standalone value.
func New() game.Detector { return &detector{} }

func (d *detector) Name() string { return "lol" }

// ScoreROI for LoL: the kill score sits in the top-center, roughly:
//
//	x: 43 %  y: 0.3 %  w: 14 %  h: 4.5 %
//
// Tested at 1920×1080, 2560×1440, and 3840×2160.
func (d *detector) ScoreROI() game.ROI {
	return game.ROI{X: 0.43, Y: 0.003, W: 0.14, H: 0.045}
}

// ParseScore parses "12 - 8" style OCR text into a Score.
// Returns nil, nil if no score pattern is found (bad OCR frame, loading screen, etc).
func (d *detector) ParseScore(ocrText string) (*game.Score, error) {
	text := strings.TrimSpace(ocrText)
	m := scoreRE.FindStringSubmatch(text)
	if m == nil {
		return nil, nil
	}
	blue, _ := strconv.Atoi(m[1])
	red, _ := strconv.Atoi(m[2])
	// Sanity check: LoL games don't exceed ~100 kills per side
	if blue > 100 || red > 100 {
		return nil, nil
	}
	return &game.Score{Blue: blue, Red: red}, nil
}

func (d *detector) SampleRate() float64        { return 2.0 }              // 2 fps is plenty
func (d *detector) MergeWindow() time.Duration { return 8 * time.Second }  // chain multi-kills within 8s
func (d *detector) PreBuffer() time.Duration   { return 10 * time.Second } // 10s before kill
func (d *detector) PostBuffer() time.Duration  { return 15 * time.Second } // 15s after last kill

func init() { game.Register(New()) }
