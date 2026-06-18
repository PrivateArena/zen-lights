package pixel

import (
	"encoding/hex"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"

	"github.com/zen-lights/zen-lights/internal/paint/engine"
	"github.com/zen-lights/zen-lights/internal/paint/sdxl"
)

// Predefined palettes
var palettes = map[string][]color.RGBA{
	"pico8": {
		{0, 0, 0, 255},       // Black
		{29, 43, 83, 255},    // Dark Blue
		{126, 37, 83, 255},   // Dark Purple
		{0, 135, 81, 255},    // Dark Green
		{171, 82, 54, 255},   // Brown
		{95, 87, 79, 255},    // Dark Grey
		{194, 195, 199, 255}, // Light Grey
		{255, 241, 232, 255}, // White
		{255, 0, 77, 255},    // Red
		{255, 163, 0, 255},   // Orange
		{255, 236, 39, 255},  // Yellow
		{0, 228, 54, 255},    // Green
		{41, 173, 255, 255},  // Blue
		{131, 118, 156, 255}, // Lavender
		{255, 119, 168, 255}, // Pink
		{255, 204, 170, 255}, // Light Peach
	},
	"gameboy": {
		{15, 56, 15, 255},     // Darkest green
		{48, 98, 48, 255},     // Dark green
		{139, 172, 15, 255},   // Light green
		{155, 188, 15, 255},   // Lightest green
	},
	"nes": {
		{124, 124, 124, 255}, {0, 0, 252, 255}, {0, 0, 188, 255}, {68, 40, 188, 255},
		{148, 0, 132, 255}, {168, 0, 32, 255}, {168, 16, 0, 255}, {136, 20, 0, 255},
		{80, 48, 0, 255}, {0, 120, 0, 255}, {0, 104, 0, 255}, {0, 88, 0, 255},
		{0, 64, 88, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255},
		{188, 188, 188, 255}, {0, 120, 248, 255}, {0, 88, 248, 255}, {104, 68, 252, 255},
		{216, 0, 204, 255}, {228, 0, 88, 255}, {248, 56, 0, 255}, {228, 92, 16, 255},
		{172, 124, 0, 255}, {0, 184, 0, 255}, {0, 168, 0, 255}, {0, 168, 68, 255},
		{0, 136, 136, 255}, {0, 0, 0, 255}, {0, 0, 0, 255}, {0, 0, 0, 255},
		{248, 248, 248, 255}, {60, 188, 252, 255}, {104, 136, 252, 255}, {152, 120, 248, 255},
		{248, 120, 248, 255}, {248, 88, 152, 255}, {248, 120, 88, 255}, {252, 160, 68, 255},
		{248, 184, 0, 255}, {184, 248, 24, 255}, {88, 216, 84, 255}, {88, 248, 152, 255},
		{0, 232, 216, 255}, {120, 120, 120, 255}, {0, 0, 0, 255}, {0, 0, 0, 255},
		{252, 252, 252, 255}, {164, 228, 252, 255}, {184, 184, 252, 255}, {216, 184, 252, 255},
		{252, 184, 252, 255}, {252, 164, 216, 255}, {252, 200, 180, 255}, {252, 224, 168, 255},
		{252, 252, 120, 255}, {216, 252, 120, 255}, {184, 252, 184, 255}, {184, 252, 216, 255},
		{0, 252, 252, 255}, {216, 216, 216, 255}, {0, 0, 0, 255}, {0, 0, 0, 255},
	},
	"c64": {
		{0, 0, 0, 255}, {255, 255, 255, 255}, {136, 0, 0, 255}, {170, 255, 238, 255},
		{204, 68, 204, 255}, {0, 204, 85, 255}, {0, 0, 170, 255}, {238, 238, 119, 255},
		{221, 136, 85, 255}, {102, 68, 0, 255}, {255, 119, 119, 255}, {51, 51, 51, 255},
		{119, 119, 119, 255}, {170, 255, 102, 255}, {0, 136, 255, 255}, {187, 187, 187, 255},
	},
}

// Engine implements engine.ImageEngine for specialized Pixel Art asset generation.
type Engine struct {
	base sdxl.Engine
}

func (e *Engine) Initialize(modelDir string, opts engine.Options) error {
	return e.base.Initialize(modelDir, opts)
}

func (e *Engine) Close() error {
	return e.base.Close()
}

func (e *Engine) Info() string {
	return "Pixel Art Generation Engine | powered by " + e.base.Info()
}

func (e *Engine) Generate(req engine.GenerateRequest) (engine.GenerateResult, error) {
	// Clean the prompt of specific palette style tags so the base model generates full color on solid white
	cleanPrompt := req.Prompt
	styleKeywords := []string{"gameboy style", "gameboy", "pico8 style", "pico8", "nes style", "nes", "c64 style", "c64"}
	for _, kw := range styleKeywords {
		cleanPrompt = strings.ReplaceAll(cleanPrompt, kw, "")
		cleanPrompt = strings.ReplaceAll(cleanPrompt, strings.Title(kw), "")
	}
	req.Prompt = strings.TrimSpace(cleanPrompt)

	// Inject style and isolation modifiers to prompt to force solid background and flat shading
	if !strings.Contains(strings.ToLower(req.Prompt), "isolated") && !strings.Contains(strings.ToLower(req.Prompt), "background") {
		req.Prompt = req.Prompt + ", isolated retro game sprite, full body, solid flat white background, clean vector, pixel art style"
	}
	negStyle := "scenery, landscape, environment, realistic, photographic, 3d render, blurry"
	if req.NegativePrompt == "" {
		req.NegativePrompt = negStyle
	} else {
		req.NegativePrompt = req.NegativePrompt + ", " + negStyle
	}


	// First run standard image generation using LCM/SDXL
	res, err := e.base.Generate(req)
	if err != nil {
		return engine.GenerateResult{}, err
	}

	// Apply post-processing pipeline
	processedPath, err := e.postProcess(res.ImagePath, req)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("pixel art post-process: %w", err)
	}

	// Update result with final path and size details
	res.ImagePath = processedPath
	if req.PixelSize > 0 {
		res.Width = req.PixelSize
		res.Height = req.PixelSize
	}
	return res, nil
}

// postProcess performs background removal, downsampling, color quantization, outlining, and double pixel cleaning.
func (e *Engine) postProcess(imgPath string, req engine.GenerateRequest) (string, error) {
	// 1. Open generated image
	f, err := os.Open(imgPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	srcImg, err := png.Decode(f)
	if err != nil {
		return "", err
	}
	f.Close() // close early so we can overwrite

	// Convert image to RGBA
	bounds := srcImg.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgbaImg := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			rgbaImg.Set(x, y, srcImg.At(bounds.Min.X+x, bounds.Min.Y+y))
		}
	}
	transparent := req.Transparent
	if !req.Transparent && req.Palette != "" {
		// Default to transparency for game assets unless explicitly disabled
		transparent = true
	}

	if transparent {
		// Remove background using dynamic color sampling and flood fill with threshold 85 to capture noise
		removeBackground(rgbaImg, 85)
	}

	// 3. Geometric Downsampling
	pixelSize := req.PixelSize
	if pixelSize <= 0 {
		pixelSize = 64 // default grid size
	}
	downscaledImg := downscale(rgbaImg, pixelSize)

	// Apply contrast adjustment (factor 1.4) to enhance readability and detail definition before palette mapping
	adjustContrast(downscaledImg, 1.4)

	// 4. Color Quantization / Palette Snapping
	paletteName := strings.ToLower(req.Palette)
	if paletteName == "" {
		paletteName = "custom" // default
	}

	if paletteName == "custom" {
		k := req.PaletteSize
		if k <= 0 {
			k = 16 // default 16 colors
		}
		quantizeKMeans(downscaledImg, k)
	} else if palette, found := palettes[paletteName]; found {
		if req.Dither {
			ditherFloydSteinberg(downscaledImg, palette)
		} else {
			snapToPalette(downscaledImg, palette)
		}
	}

	// 5. Remove tiny floating islands / speckles (clump-enforcement)
	removeIslands(downscaledImg, 3)

	// 6. Draw Outline (8-way complete outline)
	if req.Outline || (!req.Outline && req.OutlineColor != "") { // default true if not specified
		outlineColor := color.RGBA{0, 0, 0, 255} // default black
		if req.OutlineColor != "" {
			if parsed, err := parseHexColor(req.OutlineColor); err == nil {
				outlineColor = parsed
			}
		}
		downscaledImg = applyOutline(downscaledImg, outlineColor)
	}

	// 7. Clean Doubles / Staircase artifacts (smooth artwork and outline)
	if req.CleanDoubles || (!req.CleanDoubles && req.Outline) { // default true
		cleanDoubles(downscaledImg)
	}

	// 7. Save the processed low-res pixel art PNG
	outDir := filepath.Dir(imgPath)
	name := filepath.Base(imgPath)
	processedPath := filepath.Join(outDir, "px_"+name)

	fOut, err := os.Create(processedPath)
	if err != nil {
		return "", err
	}
	defer fOut.Close()

	if err := png.Encode(fOut, downscaledImg); err != nil {
		return "", err
	}

	// Optional: Delete the original high-resolution temporary file
	_ = os.Remove(imgPath)

	return processedPath, nil
}

// removeBackground removes solid backgrounds via flood fill from 4 corners and perimeter margins.
func removeBackground(img *image.RGBA, threshold int) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// 1. Clear the 3-pixel outer border to transparency to remove any AI frame/padding artifacts (if image is large enough)
	if w >= 10 && h >= 10 {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if x < 3 || x >= w-3 || y < 3 || y >= h-3 {
					img.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
				}
			}
		}
	}

	visited := make([]bool, w*h)
	type point struct{ x, y int }
	var queue []point

	colorSimilar := func(c1, c2 color.RGBA) bool {
		if c1.A == 0 || c2.A == 0 {
			return false
		}
		dr := int(c1.R) - int(c2.R)
		dg := int(c1.G) - int(c2.G)
		db := int(c1.B) - int(c2.B)
		dist := dr*dr + dg*dg + db*db
		return dist < threshold*threshold
	}

	// 2. Sample 8 points slightly inset to find the dominant background color
	insets := []point{
		{5, 5}, {w / 2, 5}, {w - 6, 5},
		{5, h / 2}, {w - 6, h / 2},
		{5, h - 6}, {w / 2, h - 6}, {w - 6, h - 6},
	}

	colorCounts := make(map[color.RGBA]int)
	for _, p := range insets {
		if p.x >= 0 && p.x < w && p.y >= 0 && p.y < h {
			c := img.RGBAAt(p.x, p.y)
			rounded := color.RGBA{
				R: uint8((int(c.R) / 10) * 10),
				G: uint8((int(c.G) / 10) * 10),
				B: uint8((int(c.B) / 10) * 10),
				A: c.A,
			}
			colorCounts[rounded]++
		}
	}

	var bgTarget color.RGBA
	maxCount := -1
	for col, count := range colorCounts {
		if count > maxCount {
			maxCount = count
			bgTarget = col
		}
	}

	if maxCount <= 0 {
		bgTarget = img.RGBAAt(0, 0)
	}

	// 3. Seed flood fill from any perimeter pixel (within 15-pixel margin) matching the dominant background color
	margin := 15
	if margin > w/2 {
		margin = w / 2
	}
	if margin > h/2 {
		margin = h / 2
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if x < margin || x >= w-margin || y < margin || y >= h-margin {
				idx := y*w + x
				if !visited[idx] && colorSimilar(img.RGBAAt(x, y), bgTarget) {
					visited[idx] = true
					queue = append(queue, point{x, y})
				}
			}
		}
	}

	// Also seed absolute corners just in case they are transparent or match background
	corners := []point{
		{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1},
	}
	for _, p := range corners {
		idx := p.y*w + p.x
		if !visited[idx] {
			visited[idx] = true
			queue = append(queue, p)
		}
	}

	// 4. Run flood fill BFS
	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		currColor := img.RGBAAt(curr.x, curr.y)

		if colorSimilar(currColor, bgTarget) || currColor.A == 0 {
			img.SetRGBA(curr.x, curr.y, color.RGBA{0, 0, 0, 0})

			neighbors := []point{
				{curr.x + 1, curr.y}, {curr.x - 1, curr.y},
				{curr.x, curr.y + 1}, {curr.x, curr.y - 1},
			}
			for _, n := range neighbors {
				if n.x >= 0 && n.x < w && n.y >= 0 && n.y < h {
					nIdx := n.y*w + n.x
					if !visited[nIdx] {
						visited[nIdx] = true
						queue = append(queue, n)
					}
				}
			}
		}
	}
}


// downscale scales an image to target size using crisp nearest-neighbor sampling.
func downscale(img image.Image, targetSize int) *image.RGBA {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	dst := image.NewRGBA(image.Rect(0, 0, targetSize, targetSize))

	for y := 0; y < targetSize; y++ {
		for x := 0; x < targetSize; x++ {
			srcX := x * srcW / targetSize
			srcY := y * srcH / targetSize

			if srcX >= srcW {
				srcX = srcW - 1
			}
			if srcY >= srcH {
				srcY = srcH - 1
			}

			c := img.At(bounds.Min.X+srcX, bounds.Min.Y+srcY)
			dst.Set(x, y, c)
		}
	}
	return dst
}

// adjustContrast modifies the contrast of an image by scaling around the middle gray level.
func adjustContrast(img *image.RGBA, factor float64) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, y)
			if c.A > 0 {
				adjust := func(val uint8) uint8 {
					v := (float64(val)-128.0)*factor + 128.0
					if v < 0 {
						return 0
					}
					if v > 255 {
						return 255
					}
					return uint8(v)
				}
				img.SetRGBA(x, y, color.RGBA{
					R: adjust(c.R),
					G: adjust(c.G),
					B: adjust(c.B),
					A: c.A,
				})
			}
		}
	}
}

// quantizeKMeans applies K-Means clustering in RGB space.
func quantizeKMeans(img *image.RGBA, k int) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	var pixels []color.RGBA
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, y)
			if c.A > 0 {
				pixels = append(pixels, c)
			}
		}
	}

	if len(pixels) == 0 {
		return
	}
	if k > len(pixels) {
		k = len(pixels)
	}

	centroids := make([]color.RGBA, k)
	r := rand.New(rand.NewSource(42))
	for i := 0; i < k; i++ {
		centroids[i] = pixels[r.Intn(len(pixels))]
	}

	distSq := perceptualDistSq

	assignments := make([]int, len(pixels))
	for iter := 0; iter < 8; iter++ {
		for i, p := range pixels {
			minDist := math.MaxInt32
			bestCentroid := 0
			for ci, c := range centroids {
				d := distSq(p, c)
				if d < minDist {
					minDist = d
					bestCentroid = ci
				}
			}
			assignments[i] = bestCentroid
		}

		sumsR := make([]int, k)
		sumsG := make([]int, k)
		sumsB := make([]int, k)
		counts := make([]int, k)
		for i, p := range pixels {
			ci := assignments[i]
			sumsR[ci] += int(p.R)
			sumsG[ci] += int(p.G)
			sumsB[ci] += int(p.B)
			counts[ci]++
		}

		for ci := 0; ci < k; ci++ {
			if counts[ci] > 0 {
				centroids[ci] = color.RGBA{
					R: uint8(sumsR[ci] / counts[ci]),
					G: uint8(sumsG[ci] / counts[ci]),
					B: uint8(sumsB[ci] / counts[ci]),
					A: 255,
				}
			}
		}
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, y)
			if c.A > 0 {
				minDist := math.MaxInt32
				bestColor := centroids[0]
				for _, centroid := range centroids {
					d := distSq(c, centroid)
					if d < minDist {
						minDist = d
						bestColor = centroid
					}
				}
				img.SetRGBA(x, y, bestColor)
			}
		}
	}
}

// snapToPalette snaps colors to the nearest color in a predefined palette.
func snapToPalette(img *image.RGBA, palette []color.RGBA) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	distSq := perceptualDistSq

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, y)
			if c.A > 0 {
				minDist := math.MaxInt32
				bestColor := palette[0]
				for _, pc := range palette {
					d := distSq(c, pc)
					if d < minDist {
						minDist = d
						bestColor = pc
					}
				}
				img.SetRGBA(x, y, bestColor)
			}
		}
	}
}

// ditherFloydSteinberg applies Floyd-Steinberg error diffusion dithering.
func ditherFloydSteinberg(img *image.RGBA, palette []color.RGBA) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	findClosestColor := func(c color.RGBA) color.RGBA {
		distSq := perceptualDistSq
		minDist := math.MaxInt32
		bestColor := palette[0]
		for _, pc := range palette {
			d := distSq(c, pc)
			if d < minDist {
				minDist = d
				bestColor = pc
			}
		}
		return bestColor
	}

	type rgbError struct{ r, g, b float32 }
	errors := make([]rgbError, w*h)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, y)
			if c.A == 0 {
				continue
			}

			idx := y*w + x
			oldR := float32(c.R) + errors[idx].r
			oldG := float32(c.G) + errors[idx].g
			oldB := float32(c.B) + errors[idx].b

			clamp := func(v float32) uint8 {
				if v < 0 {
					return 0
				}
				if v > 255 {
					return 255
				}
				return uint8(v)
			}

			clampedColor := color.RGBA{clamp(oldR), clamp(oldG), clamp(oldB), c.A}
			newColor := findClosestColor(clampedColor)

			img.SetRGBA(x, y, newColor)

			errR := oldR - float32(newColor.R)
			errG := oldG - float32(newColor.G)
			errB := oldB - float32(newColor.B)

			distribute := func(nx, ny int, factor float32) {
				if nx >= 0 && nx < w && ny >= 0 && ny < h {
					nIdx := ny*w + nx
					if img.RGBAAt(nx, ny).A > 0 {
						errors[nIdx].r += errR * factor
						errors[nIdx].g += errG * factor
						errors[nIdx].b += errB * factor
					}
				}
			}

			distribute(x+1, y, 7.0/16.0)
			distribute(x-1, y+1, 3.0/16.0)
			distribute(x, y+1, 5.0/16.0)
			distribute(x+1, y+1, 1.0/16.0)
		}
	}
}

// cleanDoubles removes double pixel corners on diagonals to thin pixel-art outlines (multi-pass).
func cleanDoubles(img *image.RGBA) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	// Perform multiple passes until the line topology stabilizes
	for pass := 0; pass < 2; pass++ {
		srcCopy := image.NewRGBA(bounds)
		copy(srcCopy.Pix, img.Pix)

		isSolid := func(x, y int) bool {
			if x < 0 || x >= w || y < 0 || y >= h {
				return false
			}
			return srcCopy.RGBAAt(x, y).A > 0
		}

		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				if !isSolid(x, y) {
					continue
				}

				// Top-Left Double Corner
				if isSolid(x-1, y) && isSolid(x, y-1) && !isSolid(x-1, y-1) {
					if !isSolid(x+1, y) && !isSolid(x, y+1) && !isSolid(x+1, y-1) && !isSolid(x-1, y+1) {
						img.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
						continue
					}
				}
				// Top-Right Double Corner
				if isSolid(x+1, y) && isSolid(x, y-1) && !isSolid(x+1, y-1) {
					if !isSolid(x-1, y) && !isSolid(x, y+1) && !isSolid(x-1, y-1) && !isSolid(x+1, y+1) {
						img.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
						continue
					}
				}
				// Bottom-Left Double Corner
				if isSolid(x-1, y) && isSolid(x, y+1) && !isSolid(x-1, y+1) {
					if !isSolid(x+1, y) && !isSolid(x, y-1) && !isSolid(x+1, y+1) && !isSolid(x-1, y-1) {
						img.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
						continue
					}
				}
				// Bottom-Right Double Corner
				if isSolid(x+1, y) && isSolid(x, y+1) && !isSolid(x+1, y+1) {
					if !isSolid(x-1, y) && !isSolid(x, y-1) && !isSolid(x-1, y+1) && !isSolid(x+1, y-1) {
						img.SetRGBA(x, y, color.RGBA{0, 0, 0, 0})
						continue
					}
				}
			}
		}
	}
}

// applyOutline adds an outline border to transparent shapes (8-way complete outline).
func applyOutline(img *image.RGBA, outlineColor color.RGBA) *image.RGBA {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()

	srcCopy := image.NewRGBA(bounds)
	copy(srcCopy.Pix, img.Pix)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if srcCopy.RGBAAt(x, y).A == 0 {
				hasSolidNeighbor := false

				// 8-way kernel check to lock down structural continuity
				for dy := -1; dy <= 1; dy++ {
					for dx := -1; dx <= 1; dx++ {
						if dx == 0 && dy == 0 {
							continue
						}
						nx, ny := x+dx, y+dy
						if nx >= 0 && nx < w && ny >= 0 && ny < h {
							neighbor := srcCopy.RGBAAt(nx, ny)
							if neighbor.A > 0 && neighbor != outlineColor {
								hasSolidNeighbor = true
								break
							}
						}
					}
					if hasSolidNeighbor {
						break
					}
				}

				if hasSolidNeighbor {
					img.SetRGBA(x, y, outlineColor)
				}
			}
		}
	}
	return img
}

// parseHexColor parses "#RRGGBB" or "RRGGBB" into color.RGBA.
func parseHexColor(s string) (color.RGBA, error) {
	s = strings.TrimPrefix(s, "#")
	if len(s) != 6 {
		return color.RGBA{}, fmt.Errorf("invalid hex length")
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return color.RGBA{}, err
	}
	return color.RGBA{b[0], b[1], b[2], 255}, nil
}

// removeIslands erases small disconnected components of solid pixels.
func removeIslands(img *image.RGBA, minSize int) {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	visited := make([]bool, w*h)

	type point struct{ x, y int }

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, y)
			if c.A == 0 {
				continue
			}

			idx := y*w + x
			if visited[idx] {
				continue
			}

			// BFS to find component
			var component []point
			var queue []point

			queue = append(queue, point{x, y})
			visited[idx] = true

			for len(queue) > 0 {
				curr := queue[0]
				queue = queue[1:]
				component = append(component, curr)

				neighbors := []point{
					{curr.x + 1, curr.y}, {curr.x - 1, curr.y},
					{curr.x, curr.y + 1}, {curr.x, curr.y - 1},
				}
				for _, n := range neighbors {
					if n.x >= 0 && n.x < w && n.y >= 0 && n.y < h {
						nIdx := n.y*w + n.x
						if !visited[nIdx] && img.RGBAAt(n.x, n.y).A > 0 {
							visited[nIdx] = true
							queue = append(queue, n)
						}
					}
				}
			}

			if len(component) < minSize {
				for _, p := range component {
					img.SetRGBA(p.x, p.y, color.RGBA{0, 0, 0, 0})
				}
			}
		}
	}
}

// perceptualDistSq calculates perceptual distance between two colors to match human luminance sensitivity.
func perceptualDistSq(c1, c2 color.RGBA) int {
	dr := int(c1.R) - int(c2.R)
	dg := int(c1.G) - int(c2.G)
	db := int(c1.B) - int(c2.B)

	meanR := (int(c1.R) + int(c2.R)) / 2
	if meanR < 128 {
		return (2 * dr * dr) + (4 * dg * dg) + (3 * db * db)
	}
	return (3 * dr * dr) + (4 * dg * dg) + (2 * db * db)
}
