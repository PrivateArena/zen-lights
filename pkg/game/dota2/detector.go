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
// Dota 2's HUD uses a slightly wider region than LoL.
var scoreRE = regexp.MustCompile(`(\d+)\s*[-–—]\s*(\d+)`)

type detector struct{}

// New returns the Dota 2 Detector.
func New() game.Detector { return &detector{} }

func (d *detector) Name() string { return "dota2" }

// ScoreROI for Dota 2: very similar to LoL but slightly wider and taller.
//
//	x: 41 %  y: 0.2 %  w: 18 %  h: 5.5 %
//
// Tested at 1920×1080 and 2560×1440.
func (d *detector) ScoreROI() game.ROI {
	return game.ROI{X: 0.41, Y: 0.002, W: 0.18, H: 0.055}
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
	// Sanity: Dota 2 games rarely exceed 80 kills per side
	if radiant > 120 || dire > 120 {
		return nil, nil
	}
	return &game.Score{Blue: radiant, Red: dire}, nil
}

func (d *detector) SampleRate() float64        { return 2.0 }
func (d *detector) MergeWindow() time.Duration { return 10 * time.Second } // Dota teamfights last longer
func (d *detector) PreBuffer() time.Duration   { return 12 * time.Second }
func (d *detector) PostBuffer() time.Duration  { return 18 * time.Second }

func init() { game.Register(New()) }
