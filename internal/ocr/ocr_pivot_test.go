package ocr

import (
	"image"
	_ "image/png"
	"os"
	"strings"
	"testing"

	_ "golang.org/x/image/webp"
)

func TestComparativeOCR(t *testing.T) {
	// Set dynamic library overrides
	libPath := "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2"
	if _, err := os.Stat(libPath); err != nil {
		t.Fatalf("libonnxruntime.so not found at %s: %v", libPath, err)
	}
	os.Setenv("ORT_SHARED_LIB_PATH", libPath)
	os.Setenv("PPOCR_MODEL_PATH", "/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_rec_infer.onnx")

	client, err := New(DefaultOptions())
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
	os.Setenv("ORT_SHARED_LIB_PATH", libPath)
	os.Setenv("PPOCR_MODEL_PATH", "/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_rec_infer.onnx")
	os.Setenv("PPOCR_DET_MODEL_PATH", "/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_det_infer.onnx")

	client, err := New(DefaultOptions())
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
