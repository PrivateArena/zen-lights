// Package imgutil provides pure-Go image manipulation helpers.
// All operations work directly on raw RGB24 byte slices or image.NRGBA
// to avoid any CGo or external library dependency in the hot path.
package imgutil

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
)

// CropRGB24 extracts a rectangular sub-region from a raw RGB24 frame buffer.
//
// src is row-major RGB24: pixel (x,y) is at src[(y*frameW + x)*3 : +3].
// roi must be within [0, frameW) × [0, frameH); it is clamped silently.
func CropRGB24(src []byte, frameW, frameH int, roi image.Rectangle) *image.NRGBA {
	roi = roi.Intersect(image.Rect(0, 0, frameW, frameH))
	out := image.NewNRGBA(image.Rect(0, 0, roi.Dx(), roi.Dy()))
	for y := roi.Min.Y; y < roi.Max.Y; y++ {
		for x := roi.Min.X; x < roi.Max.X; x++ {
			i := (y*frameW + x) * 3
			out.SetNRGBA(x-roi.Min.X, y-roi.Min.Y, color.NRGBA{
				R: src[i],
				G: src[i+1],
				B: src[i+2],
				A: 255,
			})
		}
	}
	return out
}

// ToGrayscale converts an NRGBA image to luminance-based grayscale in-place
// returning a new image (original is not modified).
func ToGrayscale(img *image.NRGBA) *image.NRGBA {
	b := img.Bounds()
	out := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := img.NRGBAAt(x, y)
			// BT.601 luma coefficients
			lum := uint8(0.299*float64(c.R) + 0.587*float64(c.G) + 0.114*float64(c.B))
			out.SetNRGBA(x, y, color.NRGBA{R: lum, G: lum, B: lum, A: 255})
		}
	}
	return out
}

// Threshold applies a hard binary threshold.
// Pixels with luminance > thresh → white (255); others → black (0).
// Works best on grayscale input (all channels equal).
func Threshold(img *image.NRGBA, thresh uint8) *image.NRGBA {
	b := img.Bounds()
	out := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := img.NRGBAAt(x, y)
			v := uint8(0)
			if c.R > thresh {
				v = 255
			}
			out.SetNRGBA(x, y, color.NRGBA{R: v, G: v, B: v, A: 255})
		}
	}
	return out
}

// InvertThreshold is like Threshold but inverts: bright pixels become black.
// Useful when the HUD text is dark-on-light rather than light-on-dark.
func InvertThreshold(img *image.NRGBA, thresh uint8) *image.NRGBA {
	b := img.Bounds()
	out := image.NewNRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := img.NRGBAAt(x, y)
			v := uint8(255)
			if c.R > thresh {
				v = 0
			}
			out.SetNRGBA(x, y, color.NRGBA{R: v, G: v, B: v, A: 255})
		}
	}
	return out
}

// ScaleUp enlarges an image by an integer factor using nearest-neighbor sampling.
// factor=3 triples both width and height — useful to improve Tesseract accuracy
// on small HUD regions where text is only 15–25px tall.
func ScaleUp(img *image.NRGBA, factor int) *image.NRGBA {
	b := img.Bounds()
	w := b.Dx() * factor
	h := b.Dy() * factor
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.SetNRGBA(x, y, img.NRGBAAt(b.Min.X+x/factor, b.Min.Y+y/factor))
		}
	}
	return out
}

// EncodePNG encodes img to a PNG byte slice in memory without touching disk.
func EncodePNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	buf.Grow(img.Bounds().Dx() * img.Bounds().Dy()) // rough pre-alloc
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
