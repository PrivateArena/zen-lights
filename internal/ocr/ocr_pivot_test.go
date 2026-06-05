package ocr

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	_ "image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	_ "golang.org/x/image/webp"

	"github.com/zen-lights/zen-lights/internal/imgutil"
)

func TestComparativeOCR(t *testing.T) {
	// Set dynamic library overrides
	libPath := "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2"
	if _, err := os.Stat(libPath); err != nil {
		t.Fatalf("libonnxruntime.so not found at %s: %v", libPath, err)
	}
	opts := DefaultOptions()
	opts.RecModelPath = "models_ocr/ch_PP-OCRv4_rec_infer.onnx"
	opts.RecVocabPath = "models_ocr/ch_dict.txt"

	client, err := New(opts)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Load integration test image
	imagePath := "/home/jang/Pictures/7727ba13ed2e0eb12011a152e9a0ff35.png"
	imgFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatalf("Failed to open test image: %v", err)
	}
	defer imgFile.Close()

	decodedImg, _, err := image.Decode(imgFile)
	if err != nil {
		t.Fatalf("Failed to decode image: %v", err)
	}

	b := decodedImg.Bounds()
	w, h := b.Dx(), b.Dy()
	frameData := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := decodedImg.At(b.Min.X+x, b.Min.Y+y)
			r, g, b, _ := c.RGBA()
			idx := (y*w + x) * 3
			frameData[idx] = uint8(r >> 8)
			frameData[idx+1] = uint8(g >> 8)
			frameData[idx+2] = uint8(b >> 8)
		}
	}

	tests := []struct {
		name     string
		rect     image.Rectangle
		expected string
	}{
		{
			name:     "Title Line",
			rect:     image.Rect(165, 32, 951, 73),
			expected: "GMKtec",
		},
		{
			name:     "Price",
			rect:     image.Rect(160, 408, 335, 455),
			expected: "2,59999",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			text, err := client.ReadRegion(frameData, w, h, tc.rect)
			if err != nil {
				t.Fatalf("ReadRegion failed: %v", err)
			}
			t.Logf("[%s] Decoded text: %q", tc.name, text)
			if !strings.Contains(text, tc.expected) {
				t.Errorf("[%s] Expected decoded text to contain %q, got %q", tc.name, tc.expected, text)
			}
		})
	}
}

func TestFullPipelineIntegration(t *testing.T) {
	// Set dynamic library overrides
	libPath := "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2"
	if _, err := os.Stat(libPath); err != nil {
		t.Fatalf("libonnxruntime.so not found at %s: %v", libPath, err)
	}
	opts := DefaultOptions()
	opts.RecModelPath = "models_ocr/ch_PP-OCRv4_rec_infer.onnx"
	opts.RecVocabPath = "models_ocr/ch_dict.txt"
	opts.DetModelPath = "models_ocr/ch_PP-OCRv4_det_infer.onnx"

	client, err := New(opts)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Load integration test image
	imagePath := "/home/jang/Pictures/7727ba13ed2e0eb12011a152e9a0ff35.png"
	imgFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatalf("Failed to open test image: %v", err)
	}
	defer imgFile.Close()

	decodedImg, _, err := image.Decode(imgFile)
	if err != nil {
		t.Fatalf("Failed to decode image: %v", err)
	}

	b := decodedImg.Bounds()
	w, h := b.Dx(), b.Dy()
	frameData := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := decodedImg.At(b.Min.X+x, b.Min.Y+y)
			r, g, b, _ := c.RGBA()
			idx := (y*w + x) * 3
			frameData[idx] = uint8(r >> 8)
			frameData[idx+1] = uint8(g >> 8)
			frameData[idx+2] = uint8(b >> 8)
		}
	}

	// Scan the entire frame layout dynamically!
	results, err := client.ReadFullFrame(frameData, w, h)
	if err != nil {
		t.Fatalf("ReadFullFrame failed: %v", err)
	}

	t.Logf("Detected %d text regions:", len(results))
	var foundTitle, foundPrice bool
	for i, res := range results {
		t.Logf("  [%d] Box: %v | Text: %q (Conf: %.4f)", i, res.Bounds, res.Text, res.Confidence)
		if strings.Contains(strings.ToLower(res.Text), "gmktec") {
			foundTitle = true
		}
		if strings.Contains(res.Text, "259999") || strings.Contains(res.Text, "2,59999") {
			foundPrice = true
		}
	}

	if !foundTitle {
		t.Errorf("Expected full pipeline to detect and read the title containing 'GMKtec', got results: %v", results)
	}
	if !foundPrice {
		t.Errorf("Expected full pipeline to detect and read the price containing '259999' or '2,59999', got results: %v", results)
	}
}

// TestMangaFullPageOCR is a diagnostic test — no assertions.
// It runs the full DBNet detection + CTC recognition pipeline on a raw manga page
// and dumps everything for human evaluation.
func TestMangaFullPageOCR(t *testing.T) {
	// --- Setup ONNX environment ---
	libPath := "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2"
	if _, err := os.Stat(libPath); err != nil {
		t.Fatalf("libonnxruntime.so not found at %s: %v", libPath, err)
	}
	opts := DefaultOptions()
	opts.RecModelPath = "models_ocr/japan_PP-OCRv4_rec_infer.onnx"
	opts.RecVocabPath = "models_ocr/japan_dict.txt"
	opts.DetModelPath = "models_ocr/ch_PP-OCRv4_det_infer.onnx"

	client, err := New(opts)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// --- Load manga test image ---
	imagePath := filepath.Join("/media/jang/home/Deve/zen-lights", "test_images", "manga-raw-chapter-101-v0-6k4q6pmm1bx81.png")
	imgFile, err := os.Open(imagePath)
	if err != nil {
		t.Fatalf("Failed to open test image: %v", err)
	}
	defer imgFile.Close()

	decodedImg, _, err := image.Decode(imgFile)
	if err != nil {
		t.Fatalf("Failed to decode image: %v", err)
	}

	b := decodedImg.Bounds()
	w, h := b.Dx(), b.Dy()
	t.Logf("Image dimensions: %dx%d", w, h)

	// Convert to RGB24 frameData
	frameData := make([]byte, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := decodedImg.At(b.Min.X+x, b.Min.Y+y)
			r, g, bl, _ := c.RGBA()
			idx := (y*w + x) * 3
			frameData[idx] = uint8(r >> 8)
			frameData[idx+1] = uint8(g >> 8)
			frameData[idx+2] = uint8(bl >> 8)
		}
	}

	// --- Phase 1: Detection timing ---
	t.Log("=== PHASE 1: Text Detection (DBNet) ===")
	detStart := time.Now()

	// Build NRGBA for detector directly
	nrgbaImg := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		srcRow := frameData[y*w*3:]
		for x := 0; x < w; x++ {
			i := x * 3
			nrgbaImg.SetNRGBA(x, y, color.NRGBA{
				R: srcRow[i], G: srcRow[i+1], B: srcRow[i+2], A: 255,
			})
		}
	}

	boxes, err := client.detEngine.LocateTextRegions(nrgbaImg)
	detDuration := time.Since(detStart)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}
	t.Logf("Detection completed in %v — found %d regions", detDuration, len(boxes))

	// --- Phase 2: Recognition timing ---
	t.Log("\n=== PHASE 2: Text Recognition (CTC) ===")
	recStart := time.Now()

	type regionResult struct {
		idx  int
		box  image.Rectangle
		text string
		conf float32
		dur  time.Duration
	}
	var recognized []regionResult

	for i, box := range boxes {
		cropStart := time.Now()
		crop := imgutil.CropRGB24(frameData, w, h, box)
		text, conf, err := client.recEngine.ProcessCrop(crop)
		cropDur := time.Since(cropStart)

		if err != nil {
			t.Logf("  [%d] Box %v — recognition error: %v", i, box, err)
			continue
		}

		recognized = append(recognized, regionResult{
			idx: i, box: box, text: text, conf: conf, dur: cropDur,
		})
	}
	recDuration := time.Since(recStart)

	// --- Results dump ---
	t.Log("\n=== RESULTS ===")
	t.Logf("Total detection time:    %v", detDuration)
	t.Logf("Total recognition time:  %v (for %d regions)", recDuration, len(boxes))
	t.Logf("Total pipeline time:     %v", detDuration+recDuration)
	t.Logf("Regions detected: %d | Recognized (conf>=0.0): %d", len(boxes), len(recognized))

	t.Log("\n--- All recognized regions (sorted by detection order) ---")
	for _, r := range recognized {
		t.Logf("  [%02d] Box(%4d,%4d)-(%4d,%4d) %3dx%3d | Conf: %.4f | Time: %6v | Text: %q",
			r.idx,
			r.box.Min.X, r.box.Min.Y, r.box.Max.X, r.box.Max.Y,
			r.box.Dx(), r.box.Dy(),
			r.conf, r.dur, r.text,
		)
	}

	// --- Save annotated debug image ---
	debugDir := filepath.Join("/media/jang/home/Deve/zen-lights", "test_images", "debug")
	os.MkdirAll(debugDir, 0o755)

	// Draw bounding boxes on a copy of the original image
	canvas := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(canvas, canvas.Bounds(), decodedImg, b.Min, draw.Src)

	red := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	green := color.RGBA{R: 0, G: 255, B: 0, A: 255}

	for _, r := range recognized {
		boxColor := red
		if r.conf >= 0.7 {
			boxColor = green
		}
		// Draw rectangle edges (2px thick)
		for t := 0; t < 2; t++ {
			for x := r.box.Min.X; x < r.box.Max.X; x++ {
				canvas.Set(x, r.box.Min.Y+t, boxColor)
				canvas.Set(x, r.box.Max.Y-1-t, boxColor)
			}
			for y := r.box.Min.Y; y < r.box.Max.Y; y++ {
				canvas.Set(r.box.Min.X+t, y, boxColor)
				canvas.Set(r.box.Max.X-1-t, y, boxColor)
			}
		}
	}

	debugPath := filepath.Join(debugDir, "manga_detection_overlay.png")
	f, err := os.Create(debugPath)
	if err != nil {
		t.Logf("Warning: failed to create debug image: %v", err)
	} else {
		png.Encode(f, canvas)
		f.Close()
		t.Logf("\nDebug overlay saved to: %s", debugPath)
	}

	// --- Summary statistics ---
	var highConf, medConf, lowConf int
	for _, r := range recognized {
		switch {
		case r.conf >= 0.7:
			highConf++
		case r.conf >= 0.4:
			medConf++
		default:
			lowConf++
		}
	}
	t.Logf("\n=== CONFIDENCE DISTRIBUTION ===")
	t.Logf("  High (≥0.7): %d", highConf)
	t.Logf("  Medium (0.4-0.7): %d", medConf)
	t.Logf("  Low (<0.4): %d", lowConf)

	// Note about model usage
	t.Log("\n✅ Using japan_PP-OCRv4 model (Japanese).")
	t.Log("   Hiragana/Katakana and Kanji should be recognized correctly.")

	// Suppress unused import
	_ = fmt.Sprint
}
