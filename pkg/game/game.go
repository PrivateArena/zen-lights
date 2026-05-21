// Package game defines the Detector interface and shared types.
// To add a new game, implement Detector and call game.Register() in your init().
package game

import (
	"image"
	"time"
)

// ROI defines a region of interest as relative coordinates in [0.0, 1.0].
// This ensures the detector works across all resolutions (1080p, 1440p, 4K).
type ROI struct {
	X, Y, W, H float64
}

// ToImageRect converts relative ROI coordinates to absolute pixel coordinates.
func (r ROI) ToImageRect(frameW, frameH int) image.Rectangle {
	x0 := int(r.X * float64(frameW))
	y0 := int(r.Y * float64(frameH))
	x1 := x0 + int(r.W*float64(frameW))
	y1 := y0 + int(r.H*float64(frameH))
	return image.Rect(x0, y0, x1, y1)
}

// Score represents the current kill tally for both sides.
type Score struct {
	Blue int // left-side team (Radiant, Blue side, etc.)
	Red  int // right-side team (Dire, Red side, etc.)
}

// Equal returns true when both scores match.
func (s Score) Equal(other Score) bool {
	return s.Blue == other.Blue && s.Red == other.Red
}

// KillEvent marks a moment where a kill was detected by a score change.
type KillEvent struct {
	At          time.Duration
	ScoreBefore Score
	ScoreAfter  Score
}

// Delta returns total new kills in this event (handles multi-kills in one frame gap).
func (e KillEvent) Delta() int {
	db := e.ScoreAfter.Blue - e.ScoreBefore.Blue
	dr := e.ScoreAfter.Red - e.ScoreBefore.Red
	if db < 0 {
		db = 0
	}
	if dr < 0 {
		dr = 0
	}
	return db + dr
}

// Detector is the interface each supported game must implement.
// Register your implementation via game.Register() inside an init() function.
type Detector interface {
	// Name returns the game's unique identifier (e.g. "lol", "dota2").
	Name() string

	// ScoreROI returns the relative bounding box of the kill score HUD element.
	ScoreROI() ROI

	// ParseScore extracts a Score from raw OCR text returned by the OCR engine.
	// Return nil, nil when no recognizable score is present in the text.
	ParseScore(ocrText string) (*Score, error)

	// SampleRate is how many frames per second to decode from the video.
	// 2–4 fps is enough for kill scoring; higher wastes CPU.
	SampleRate() float64

	// MergeWindow is the max gap between two kill events before they become
	// separate highlight segments rather than one chained multi-kill segment.
	MergeWindow() time.Duration

	// PreBuffer is how much video to include before the first kill in a segment.
	PreBuffer() time.Duration

	// PostBuffer is how much video to include after the last kill in a segment.
	PostBuffer() time.Duration
}
