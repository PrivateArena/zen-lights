package ocr

import (
	"image"
	_ "image/png"
	"math"
	"os"
	"strings"
	"testing"

	_ "golang.org/x/image/webp"

	ort "github.com/yalue/onnxruntime_go"
)

func TestComparativeOCR(t *testing.T) {
	// Set dynamic library override
	libPath := "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2"
	if _, err := os.Stat(libPath); err != nil {
		t.Fatalf("libonnxruntime.so not found at %s: %v", libPath, err)
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("Failed to initialize ONNX Runtime environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Load model
	modelPath := "/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_rec_infer.onnx"
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"x"},
		[]string{"softmax_11.tmp_0"},
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to create ONNX session: %v", err)
	}
	defer session.Destroy()

	// Load vocabulary keys
	vocabBytes, err := os.ReadFile("/media/jang/home/Deve/zen-lights/internal/ocr/ppocr_keys_v4.txt")
	if err != nil {
		t.Fatalf("Failed to read vocab file: %v", err)
	}
	keys := strings.Split(string(vocabBytes), "\n")

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

	nrgbaImg, ok := decodedImg.(*image.NRGBA)
	if !ok {
		// Convert to NRGBA if not already
		b := decodedImg.Bounds()
		nrgbaImg = image.NewNRGBA(b)
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				nrgbaImg.Set(x, y, decodedImg.At(x, y))
			}
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
			crop := image.NewNRGBA(image.Rect(0, 0, tc.rect.Dx(), tc.rect.Dy()))
			for y := tc.rect.Min.Y; y < tc.rect.Max.Y; y++ {
				for x := tc.rect.Min.X; x < tc.rect.Max.X; x++ {
					crop.SetNRGBA(x-tc.rect.Min.X, y-tc.rect.Min.Y, nrgbaImg.NRGBAAt(x, y))
				}
			}

			// Preprocessing: Resize to height 48 and keep aspect ratio
			h := 48
			w := int(math.Ceil(float64(h) * float64(tc.rect.Dx()) / float64(tc.rect.Dy())))
			resized := resizeBilinear(crop, w, h)

			// Build flat BGR normalised input array [1, 3, 48, w]
			inputData := make([]float32, 1*3*h*w)
			idx := 0
			// Blue channel
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					inputData[idx] = float32(resized.NRGBAAt(x, y).B)/127.5 - 1.0
					idx++
				}
			}
			// Green channel
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					inputData[idx] = float32(resized.NRGBAAt(x, y).G)/127.5 - 1.0
					idx++
				}
			}
			// Red channel
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					inputData[idx] = float32(resized.NRGBAAt(x, y).R)/127.5 - 1.0
					idx++
				}
			}

			// Create inputs and outputs tensors
			inputShape := ort.NewShape(1, 3, int64(h), int64(w))
			inputTensor, err := ort.NewTensor(inputShape, inputData)
			if err != nil {
				t.Fatalf("Failed to create input tensor: %v", err)
			}
			defer inputTensor.Destroy()

			inputs := []ort.Value{inputTensor}
			outputs := []ort.Value{nil}

			// Run Inference
			err = session.Run(inputs, outputs)
			if err != nil {
				t.Fatalf("Failed to run session: %v", err)
			}

			// Process output
			outTensor, ok := outputs[0].(*ort.Tensor[float32])
			if !ok {
				t.Fatalf("Failed to cast output to float32 tensor")
			}

			outData := outTensor.GetData()
			outShape := outTensor.GetShape()
			t.Logf("[%s] Output shape: %v", tc.name, outShape)

			seqLen := int(outShape[1])
			numClasses := int(outShape[2])

			// Decode using CTC greedy decoding from the main implementation
			text, confidence := ctcGreedyDecode(outData, seqLen, numClasses, keys)
			t.Logf("[%s] Decoded text: %q (confidence: %.4f)", tc.name, text, confidence)
			if !strings.Contains(text, tc.expected) {
				t.Errorf("[%s] Expected decoded text to contain %q, got %q", tc.name, tc.expected, text)
			}
		})
	}
}

func TestClientIntegration(t *testing.T) {
	// Set dynamic library override
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

	// Convert image to raw RGB24 bytes
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

	// Read Title region X: 165, Y: 32, W: 786, H: 41
	rect := image.Rect(165, 32, 951, 73)
	text, err := client.ReadRegion(frameData, w, h, rect)
	if err != nil {
		t.Fatalf("ReadRegion failed: %v", err)
	}
	t.Logf("Client read: %q", text)

	if !strings.Contains(text, "GMKtec") {
		t.Errorf("Expected client read to contain 'GMKtec', got %q", text)
	}
}

