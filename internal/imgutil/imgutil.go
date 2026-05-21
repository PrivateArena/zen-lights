// Package imgutil provides pure-Go image manipulation helpers.
// All operations work directly on raw RGB24 byte slices to avoid any CGo or
// external library dependency in the hot path.
package imgutil

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
)

// GrayImage is an alias so callers can distinguish grayscale-only images from
// full-colour ones at the type level, making threshold pipelines self-documenting.
type GrayImage = image.NRGBA

// CropRGB24 extracts a rectangular sub-region from a raw RGB24 frame buffer.
//
// src is row-major RGB24: pixel (x,y) is at src[(y*frameW + x)*3 : +3].
// roi is clamped to the frame bounds silently.
func CropRGB24(src []byte, frameW, frameH int, roi image.Rectangle) *image.NRGBA {
	roi = roi.Intersect(image.Rect(0, 0, frameW, frameH))
	out := image.NewNRGBA(image.Rect(0, 0, roi.Dx(), roi.Dy()))
	for y := roi.Min.Y; y < roi.Max.Y; y++ {
		srcRow := src[y*frameW*3:]
		for x := roi.Min.X; x < roi.Max.X; x++ {
			i := x * 3
			out.SetNRGBA(x-roi.Min.X, y-roi.Min.Y, color.NRGBA{
				R: srcRow[i],
				G: srcRow[i+1],
				B: srcRow[i+2],
				A: 255,
			})
		}
	}
	return out
}

// ToGrayscale converts an NRGBA image to luminance-based grayscale.
// Uses BT.601 luma coefficients. Returns a new image; input is unchanged.
func ToGrayscale(img *image.NRGBA) *GrayImage {
	b := img.Bounds()
	out := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := img.NRGBAAt(x, y)
			lum := uint8(0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B))
			out.SetNRGBA(x, y, color.NRGBA{R: lum, G: lum, B: lum, A: 255})
		}
	}
	return out
}

// Threshold applies a hard binary threshold.
// Pixels with luminance > thresh → white (255); others → black (0).
// Best for bright text on a dark HUD background (LoL, most MOBAs).
func Threshold(img *GrayImage, thresh uint8) *GrayImage {
	return threshold(img, thresh, false)
}

// InvertThreshold is like Threshold but inverted: bright pixels become black.
// Use when the HUD has dark text on a light background.
func InvertThreshold(img *GrayImage, thresh uint8) *GrayImage {
	return threshold(img, thresh, true)
}

func threshold(img *GrayImage, thresh uint8, invert bool) *GrayImage {
	b := img.Bounds()
	out := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			lum := img.NRGBAAt(x, y).R
			bright := lum > thresh
			if invert {
				bright = !bright
			}
			v := uint8(0)
			if bright {
				v = 255
			}
			out.SetNRGBA(x, y, color.NRGBA{R: v, G: v, B: v, A: 255})
		}
	}
	return out
}

// AdaptiveThreshold applies Sauvola-style local thresholding.
// It divides the image into tiles and thresholds each tile independently,
// which handles HUDs with uneven lighting or semi-transparent backgrounds
// far better than a single global threshold.
//
// windowRadius controls tile half-size (default 8 → 17×17 px windows).
// k is the sensitivity parameter (0.2 is a good starting point).
func AdaptiveThreshold(img *GrayImage) *GrayImage {
	const windowRadius = 8
	const k = 0.2
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := image.NewNRGBA(b)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			// Compute local mean and standard deviation in the window
			x0 := max(0, x-windowRadius)
			y0 := max(0, y-windowRadius)
			x1 := min(w-1, x+windowRadius)
			y1 := min(h-1, y+windowRadius)

			var sum, sumSq float64
			count := 0
			for ny := y0; ny <= y1; ny++ {
				for nx := x0; nx <= x1; nx++ {
					v := float64(img.NRGBAAt(b.Min.X+nx, b.Min.Y+ny).R)
					sum += v
					sumSq += v * v
					count++
				}
			}
			mean := sum / float64(count)
			variance := sumSq/float64(count) - mean*mean
			stddev := 0.0
			if variance > 0 {
				stddev = sqrt(variance)
			}
			localThresh := mean * (1 + k*(stddev/128.0-1))

			lum := float64(img.NRGBAAt(b.Min.X+x, b.Min.Y+y).R)
			v := uint8(0)
			if lum > localThresh {
				v = 255
			}
			out.SetNRGBA(b.Min.X+x, b.Min.Y+y, color.NRGBA{R: v, G: v, B: v, A: 255})
		}
	}
	return out
}

// ScaleUp enlarges an image by an integer factor using nearest-neighbor.
// factor=3 triples both width and height — improves Tesseract accuracy on
// small HUD regions where text is only 15–25px tall.
func ScaleUp(img *image.NRGBA, factor int) *image.NRGBA {
	b := img.Bounds()
	ow := b.Dx() * factor
	oh := b.Dy() * factor
	out := image.NewNRGBA(image.Rect(0, 0, ow, oh))
	for y := 0; y < oh; y++ {
		for x := 0; x < ow; x++ {
			out.SetNRGBA(x, y, img.NRGBAAt(b.Min.X+x/factor, b.Min.Y+y/factor))
		}
	}
	return out
}

// EncodePNG encodes img to a PNG byte slice in memory without touching disk.
func EncodePNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	buf.Grow(img.Bounds().Dx() * img.Bounds().Dy())
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Convenience wrappers to avoid math import in this package.
func sqrt(x float64) float64 {
	// Newton-Raphson, sufficient for our small stddev values
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 20; i++ {
		z -= (z*z - x) / (2 * z)
	}
	return z
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
