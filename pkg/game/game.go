// Package game defines the Detector interface and shared types.
// To add a new game, implement Detector and call game.Register() in your init().
package game

import (
	"image"
	"time"
)

// ROI defines a region of interest as relative coordinates in [0.0, 1.0].
// Using relative coordinates means the detector works across all resolutions
// (1080p, 1440p, 4K) without any per-resolution tuning.
type ROI struct {
	X, Y, W, H float64
}

// ToImageRect converts relative ROI coordinates to absolute pixel coordinates
// for a frame with the given dimensions.
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

// Equal returns true when both scores match exactly.
func (s Score) Equal(other Score) bool {
	return s.Blue == other.Blue && s.Red == other.Red
}

// KillEvent marks a moment where the score changed, indicating one or more kills.
type KillEvent struct {
	At          time.Duration
	ScoreBefore Score
	ScoreAfter  Score
}

// Delta returns total new kills in this event.
// Handles multi-kills (Pentakill etc.) that appear in a single frame gap.
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

// OCRConfig carries OCR preprocessing parameters a Detector can override.
// The zero value falls back to the global defaults (bright-text-on-dark,
// 3× scale, threshold=110).
type OCRConfig struct {
	// ScaleFactor overrides the upscale multiplier (3 or 4). Zero = use default.
	ScaleFactor int

	// ThresholdValue overrides the binarisation cutoff [0–255]. Zero = use default.
	ThresholdValue uint8

	// InvertThreshold switches to dark-on-light mode (e.g. if HUD text is dark).
	InvertThreshold bool

	// AdaptiveThreshold enables local (Sauvola-style) thresholding, which
	// handles uneven backgrounds better than a global cutoff. Overrides Invert.
	AdaptiveThreshold bool
}

// Detector is the interface each supported game must implement.
// Register your implementation via game.Register() inside an init() function.
type Detector interface {
	// Name returns the game's unique identifier (e.g. "lol", "dota2").
	Name() string

	// ScoreROI returns the relative bounding box of the kill score HUD element.
	// Measure in a representative screenshot and convert to fractions:
	//   X = pixelX / frameWidth, W = roiWidth / frameWidth (similarly for Y/H).
	ScoreROI() ROI

	// ParseScore extracts a Score from raw OCR text.
	// Return nil, nil when no recognisable score is present.
	ParseScore(ocrText string) (*Score, error)

	// SampleRate is how many frames per second to decode from the video.
	// 2–4 fps is enough for kill scoring; higher costs proportionally more CPU.
	SampleRate() float64

	// MergeWindow is the maximum gap between two kill events before they are
	// split into separate highlight segments.
	MergeWindow() time.Duration

	// PreBuffer is how much video to include before the first kill in a segment.
	PreBuffer() time.Duration

	// PostBuffer is how much video to include after the last kill in a segment.
	PostBuffer() time.Duration
}

// OCRConfigProvider is an optional interface a Detector can implement to
// override the default OCR preprocessing parameters.
// The pipeline checks for it at runtime via a type assertion.
type OCRConfigProvider interface {
	OCRConfig() OCRConfig
}
