// Package ocr wraps Tesseract (via gosseract) and exposes a single method
// optimised for reading kill-score digits from cropped HUD regions.
//
// Requires Tesseract to be installed:
//
//	macOS:  brew install tesseract
//	Ubuntu: apt install tesseract-ocr
package ocr

import (
	"fmt"
	"image"
	"strings"

	"github.com/otiai10/gosseract/v2"
	"github.com/zen-lights/zen-lights/internal/imgutil"
	"github.com/zen-lights/zen-lights/pkg/game"
)

// Options configures OCR preprocessing.
// Build one from a game.OCRConfig with OptionsFromConfig.
type Options struct {
	scaleFactor       int
	thresholdValue    uint8
	invertThreshold   bool
	adaptiveThreshold bool
}

// DefaultOptions returns sensible defaults for bright-text-on-dark MOBAs
// (LoL, Dota 2, CS2, Valorant, etc.).
func DefaultOptions() Options {
	return Options{
		scaleFactor:    3,
		thresholdValue: 110,
	}
}

// OptionsFromConfig builds Options from a game.OCRConfig, applying defaults
// for any zero-value fields.
func OptionsFromConfig(c game.OCRConfig) Options {
	o := DefaultOptions()
	if c.ScaleFactor > 0 {
		o.scaleFactor = c.ScaleFactor
	}
	if c.ThresholdValue > 0 {
		o.thresholdValue = c.ThresholdValue
	}
	o.invertThreshold = c.InvertThreshold
	o.adaptiveThreshold = c.AdaptiveThreshold
	return o
}

// Client holds a long-lived Tesseract session configured for digit reading.
// Create one per pipeline run and defer Close().
// Not safe for concurrent use — create one per goroutine if parallelism needed.
type Client struct {
	tess *gosseract.Client
	opts Options
}

// New creates a Tesseract client with the given options.
//
// Tesseract is configured with:
//   - Character whitelist: digits, space, dash, en-dash
//   - PSM 7: single text line
//   - OEM 3: LSTM + legacy fallback
func New(opts Options) (*Client, error) {
	t := gosseract.NewClient()

	if err := t.SetVariable("tessedit_char_whitelist", "0123456789 -–"); err != nil {
		t.Close()
		return nil, fmt.Errorf("set char whitelist: %w", err)
	}
	t.SetPageSegMode(gosseract.PSM_SINGLE_LINE)

	return &Client{tess: t, opts: opts}, nil
}

// Close releases the underlying Tesseract instance. Always defer this.
func (c *Client) Close() { c.tess.Close() }

// ReadRegion crops roi from the raw RGB24 frame, preprocesses it, and returns
// the recognised text.
//
// Preprocessing pipeline:
//  1. Crop the ROI (pure Go)
//  2. Convert to grayscale (BT.601 luma)
//  3. Scale up (default 3×) — Tesseract performs better on ≥40px text
//  4. Binary threshold (direction / adaptive mode from Options)
//  5. PNG-encode in-memory → Tesseract
//
// Returns empty string (not an error) when no text can be read.
func (c *Client) ReadRegion(frameData []byte, frameW, frameH int, roi image.Rectangle) (string, error) {
	text, _, err := c.ReadRegionWithDiagnostics(frameData, frameW, frameH, roi, false)
	return text, err
}

// ReadRegionWithDiagnostics is like ReadRegion but optionally also returns the
// preprocessed PNG for debugging ROI/threshold tuning.
// Pass dumpPNG=false in production to skip the extra allocation.
func (c *Client) ReadRegionWithDiagnostics(
	frameData []byte, frameW, frameH int, roi image.Rectangle, dumpPNG bool,
) (text string, debugPNG []byte, err error) {
	// 1. Crop
	crop := imgutil.CropRGB24(frameData, frameW, frameH, roi)

	// 2. Grayscale
	gray := imgutil.ToGrayscale(crop)

	// 3. Scale up
	scaled := imgutil.ScaleUp(gray, c.opts.scaleFactor)

	// 4. Threshold
	var thresh *imgutil.GrayImage
	switch {
	case c.opts.adaptiveThreshold:
		thresh = imgutil.AdaptiveThreshold(scaled)
	case c.opts.invertThreshold:
		thresh = imgutil.InvertThreshold(scaled, c.opts.thresholdValue)
	default:
		thresh = imgutil.Threshold(scaled, c.opts.thresholdValue)
	}

	// 5. PNG encode
	pngBytes, encErr := imgutil.EncodePNG(thresh)
	if encErr != nil {
		return "", nil, fmt.Errorf("png encode: %w", encErr)
	}
	if dumpPNG {
		debugPNG = pngBytes
	}

	if err := c.tess.SetImageFromBytes(pngBytes); err != nil {
		return "", debugPNG, fmt.Errorf("tesseract set image: %w", err)
	}
	raw, tessErr := c.tess.Text()
	if tessErr != nil {
		return "", debugPNG, fmt.Errorf("tesseract text: %w", tessErr)
	}
	return strings.TrimSpace(raw), debugPNG, nil
}
