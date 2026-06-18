package pixel

import (
	"image"
	"image/color"
	"testing"
)

func TestDownscale(t *testing.T) {
	// Create a 10x10 test image with a pattern
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			if x < 5 {
				img.SetRGBA(x, y, color.RGBA{255, 0, 0, 255}) // Red
			} else {
				img.SetRGBA(x, y, color.RGBA{0, 255, 0, 255}) // Green
			}
		}
	}

	// Downscale to 2x2
	downscaled := downscale(img, 2)
	bounds := downscaled.Bounds()
	if bounds.Dx() != 2 || bounds.Dy() != 2 {
		t.Fatalf("expected 2x2 image, got %dx%d", bounds.Dx(), bounds.Dy())
	}

	// Nearest neighbor check
	// x=0 maps to x=0 (Red)
	// x=1 maps to x=5 (Green)
	c00 := downscaled.RGBAAt(0, 0)
	c10 := downscaled.RGBAAt(1, 0)

	if c00.R != 255 || c00.G != 0 {
		t.Errorf("expected pixel (0,0) to be Red, got %v", c00)
	}
	if c10.R != 0 || c10.G != 255 {
		t.Errorf("expected pixel (1,0) to be Green, got %v", c10)
	}
}

func TestRemoveBackground(t *testing.T) {
	// Create 5x5 image with a white background and a red center
	img := image.NewRGBA(image.Rect(0, 0, 5, 5))
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			img.SetRGBA(x, y, color.RGBA{255, 255, 255, 255}) // White background
		}
	}
	img.SetRGBA(2, 2, color.RGBA{255, 0, 0, 255}) // Red center

	// Remove background (flood fill from corners)
	removeBackground(img, 10)

	// Corners should be transparent
	if img.RGBAAt(0, 0).A != 0 {
		t.Error("expected corner (0,0) to be transparent")
	}
	if img.RGBAAt(4, 4).A != 0 {
		t.Error("expected corner (4,4) to be transparent")
	}
	// Center should remain solid red
	center := img.RGBAAt(2, 2)
	if center.R != 255 || center.A != 255 {
		t.Errorf("expected center to remain solid red, got %v", center)
	}
}

func TestQuantizeKMeans(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	// 8 red-ish pixels and 8 green-ish pixels
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if x < 2 {
				img.SetRGBA(x, y, color.RGBA{250, 10, 10, 255})
			} else {
				img.SetRGBA(x, y, color.RGBA{10, 240, 10, 255})
			}
		}
	}

	// Quantize to 2 colors
	quantizeKMeans(img, 2)

	// Get colors at (0,0) and (2,0)
	c1 := img.RGBAAt(0, 0)
	c2 := img.RGBAAt(2, 0)

	if c1 == c2 {
		t.Errorf("expected 2 distinct clusters, got both same: %v", c1)
	}
}

func TestApplyOutline(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 3, 3))
	// Center is solid red, rest is transparent
	img.SetRGBA(1, 1, color.RGBA{255, 0, 0, 255})

	outlineColor := color.RGBA{0, 0, 0, 255}
	img = applyOutline(img, outlineColor)

	// The neighbors of (1,1) should now have the outline color
	if img.RGBAAt(0, 1) != outlineColor {
		t.Errorf("expected outline at (0,1), got %v", img.RGBAAt(0, 1))
	}
	if img.RGBAAt(1, 0) != outlineColor {
		t.Errorf("expected outline at (1,0), got %v", img.RGBAAt(1, 0))
	}
	// Corners should also have outline color under 8-way complete outline
	if img.RGBAAt(0, 0) != outlineColor {
		t.Errorf("expected outline at corner (0,0) under 8-way mode, got %v", img.RGBAAt(0, 0))
	}
}

func TestCleanDoubles(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	// Create a diagonal L-shape double:
	// . O . .
	// O O . .
	// . . . .
	// . . . .
	img.SetRGBA(1, 0, color.RGBA{0, 0, 0, 255})
	img.SetRGBA(0, 1, color.RGBA{0, 0, 0, 255})
	img.SetRGBA(1, 1, color.RGBA{0, 0, 0, 255})

	cleanDoubles(img)

	// The corner double (1,1) should have been removed (turned transparent)
	if img.RGBAAt(1, 1).A != 0 {
		t.Error("expected corner double at (1,1) to be cleaned (made transparent)")
	}
}

func TestRemoveIslands(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 5, 5))
	// Create a large 3x3 component and a small 1x1 island
	// Large component (center of 3x3)
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			img.SetRGBA(x, y, color.RGBA{255, 255, 255, 255})
		}
	}
	// Floating single island at (0,0)
	img.SetRGBA(0, 0, color.RGBA{255, 255, 255, 255})

	// Remove islands smaller than 3 pixels
	removeIslands(img, 3)

	// Floating single pixel at (0,0) should be removed
	if img.RGBAAt(0, 0).A != 0 {
		t.Error("expected floating island at (0,0) to be removed")
	}

	// Large component should remain intact
	if img.RGBAAt(2, 2).A != 255 {
		t.Error("expected main component to remain solid")
	}
}
