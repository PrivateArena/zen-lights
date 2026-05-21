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

	"github.com/otiai10/gosseract/v2"
	"github.com/zen-lights/zen-lights/internal/imgutil"
)

// Client holds a long-lived Tesseract session configured for digit reading.
// Create one per pipeline run and defer Close().
// Not safe for concurrent use from multiple goroutines — create one per goroutine
// if you need parallelism.
type Client struct {
	tess *gosseract.Client
}

// New creates a Tesseract client pre-configured for HUD kill-score OCR:
//   - Character whitelist: digits, space, dash (typical score format "12 - 8")
//   - PSM 7: treat the image as a single text line
//   - OEM 3: default engine (LSTM + legacy fallback)
func New() (*Client, error) {
	t := gosseract.NewClient()

	if err := t.SetVariable("tessedit_char_whitelist", "0123456789 -–"); err != nil {
		t.Close()
		return nil, fmt.Errorf("set char whitelist: %w", err)
	}
	// PSM_SINGLE_LINE (7) — the score is always a single horizontal line
	t.SetPageSegMode(gosseract.PSM_SINGLE_LINE)

	return &Client{tess: t}, nil
}

// Close releases the underlying Tesseract instance. Always defer this.
func (c *Client) Close() { c.tess.Close() }

// ReadRegion crops roi from the raw RGB24 frame, preprocesses it for maximum
// OCR accuracy, and returns the recognised text.
//
// Preprocessing steps:
//  1. Crop the ROI (pure Go, zero-copy arithmetic on the source slice)
//  2. Convert to grayscale (BT.601 luma)
//  3. Scale up 3× with nearest-neighbor (Tesseract performs better on ≥40px text)
//  4. Binary threshold (bright HUD text on dark background)
//  5. PNG-encode in-memory → Tesseract
//
// Returns an empty string (not an error) when no text can be read.
func (c *Client) ReadRegion(frameData []byte, frameW, frameH int, roi image.Rectangle) (string, error) {
	// 1. Crop
	crop := imgutil.CropRGB24(frameData, frameW, frameH, roi)

	// 2. Grayscale
	gray := imgutil.ToGrayscale(crop)

	// 3. Scale up 3× — HUD text is typically only 18–30px tall
	scaled := imgutil.ScaleUp(gray, 3)

	// 4. Threshold — assumes bright digits on a dark HUD background.
	//    Adjust threshold value or swap to InvertThreshold if your game differs.
	thresh := imgutil.Threshold(scaled, 100)

	// 5. Encode to PNG in memory (no disk I/O)
	pngBytes, err := imgutil.EncodePNG(thresh)
	if err != nil {
		return "", fmt.Errorf("png encode: %w", err)
	}

	if err := c.tess.SetImageFromBytes(pngBytes); err != nil {
		return "", fmt.Errorf("tesseract set image: %w", err)
	}

	text, err := c.tess.Text()
	if err != nil {
		return "", fmt.Errorf("tesseract text: %w", err)
	}
	return text, nil
}
