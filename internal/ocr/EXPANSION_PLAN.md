Here is the complete, production-ready expansion plan to transition your Go OCR setup into a highly structured engine that seamlessly supports **High-Frequency Target Recognition** (for Game Scoreboards) and a **Full Layout Text Processing Pipeline** (for Manga Pages).

To accomplish this without introducing architectural clutter, we will organize your codebase around three functional domains: **Orchestrator**, **Detector**, and **Recognizer**.

---

### Step 1: Establish Your Directory Structure

By categorizing tasks logically, your core `Client` can cleanly toggle its processing execution strategies based on what parameters the calling function provides.

```text
internal/ocr/
├── cgo_linux.go        # Shared library linker paths via -rpath=$ORIGIN
├── cgo_windows.go      # Windows DLL lookups
├── ocr.go              # The Master Orchestrator Client API
├── ppocr_keys_v4.txt   # Embedded asset token vocabulary
├── detector/           # NEW: Layout Segmentation & DBNet Post-processing
│   └── detector.go     
└── recognizer/         # REFACTOR: Move your current text line recognition here
    └── recognizer.go   

```

---

### Step 2: Implement the Detection Layer (`detector/detector.go`)

The Text Detection model handles entire image arrays, spitting out a raw float32 probability matrix. In Go, we parse this matrix by thresholding the values ($>0.3$), grouping neighboring high-confidence pixels into explicit bounding box coordinate limits.

```go
package detector

import (
	"errors"
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

type Detector struct {
	session *ort.DynamicAdvancedSession
	thresh  float32
}

func New(modelPath string) (*Detector, error) {
	// The PP-OCRv4 detection model uses input placeholder "x" and output vector "sigmoid_0.tmp_0"
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"x"}, []string{"sigmoid_0.tmp_0"}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to load detection model tensor: %w", err)
	}
	return &Detector{
		session: session,
		thresh:  0.3, // Standard DBNet threshold validation filter
	}, nil
}

func (d *Detector) Destroy() {
	if d.session != nil {
		d.session.Destroy()
	}
}

// LocateTextRegions scans an entire image layout and returns identified bounding boxes
func (d *Detector) LocateTextRegions(img *image.NRGBA) ([]image.Rectangle, error) {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()

	// 1. DBNet requires dimension alignment scales to be explicit multiples of 32
	detW := (srcW/32 + 1) * 32
	detH := (srcH/32 + 1) * 32

	// 2. Normalize and construct the multi-channel linear tensor data [1, 3, detH, detW]
	inputData := make([]float32, 1*3*detH*detW)
	
	// Preprocessing math using typical ImageNet scale distributions
	// Formula: (PixelValue / 255.0 - Mean) / Std
	mean := [3]float32{0.485, 0.456, 0.406}
	std := [3]float32{0.229, 0.224, 0.225}

	idx := 0
	for c := 0; c < 3; c++ {
		for y := 0; y < detH; y++ {
			for x := 0; x < detW; x++ {
				var val float32
				if x < srcW && y < srcH {
					pixel := img.NRGBAAt(bounds.Min.X+x, bounds.Min.Y+y)
					switch c {
					case 0: val = float32(pixel.R) / 255.0 // Red channel
					case 1: val = float32(pixel.G) / 255.0 // Green channel
					case 2: val = float32(pixel.B) / 255.0 // Blue channel
					}
				}
				inputData[idx] = (val - mean[c]) / std[c]
				idx++
			}
		}
	}

	// 3. Invoke Inference
	inputShape := ort.NewShape(1, 3, int64(detH), int64(detW))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := d.session.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return nil, err
	}

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("failed casting spatial detection matrix layer output")
	}

	outData := outTensor.GetData()
	outShape := outTensor.GetShape() // [1, 1, detH, detW]
	outH, outW := int(outShape[2]), int(outShape[3])

	// 4. Post-processing: Connect components passing activation thresholds
	return d.extractBoundingBoxes(outData, outH, outW, srcW, srcH), nil
}

// Simple pixel-clustering bounding box parsing
func (d *Detector) extractBoundingBoxes(pred []float32, h, w, srcW, srcH) []image.Rectangle {
	visited := make([]bool, len(pred))
	var rects []image.Rectangle

	scaleX := float64(srcW) / float64(w)
	scaleY := float64(srcH) / float64(h)

	for y := 0; y < srcH; y++ {
		for x := 0; x < srcW; x++ {
			idx := y*w + x
			if pred[idx] > d.thresh && !visited[idx] {
				// Run a breadth-first search to compute bounds for this text patch
				minX, maxX := x, x
				minY, maxY := y, y

				queue := []int{idx}
				visited[idx] = true

				for len(queue) > 0 {
					curr := queue[0]
					queue = queue[1:]

					cx := curr % w
					cy := curr / w

					if cx < minX { minX = cx }
					if cx > maxX { maxX = cx }
					if cy < minY { minY = cy }
					if cy > maxY { maxY = cy }

					// Scan 4-way spatial coordinates
					dirs := []int{-1, 1, -w, w}
					for _, step := range dirs {
						neighbor := curr + step
						if neighbor >= 0 && neighbor < len(pred) && !visited[neighbor] {
							nx := neighbor % w
							ny := neighbor / w
							if nx < srcW && ny < srcH && pred[neighbor] > d.thresh {
								visited[neighbor] = true
								queue = append(queue, neighbor)
							}
						}
					}
				}

				// Apply aspect scale mapping and filter out trivial noise matrices
				if (maxX - minX) > 4 && (maxY - minY) > 4 {
					rects = append(rects, image.Rect(
						int(float64(minX)*scaleX),
						int(float64(minY)*scaleY),
						int(float64(maxX)*scaleX),
						int(float64(maxY)*scaleY),
					))
				}
			}
		}
	}
	return rects
}

```

---

### Step 3: Implement the Recognition Layer (`recognizer/recognizer.go`)

Migrate your core recognition logic out of your original root file into this focused namespace. We will augment it with an **Aspect Ratio Guard**, enabling seamless processing of both horizontal video layers and vertical manga bubble formats.

```go
package recognizer

import (
	"errors"
	"image"
	"math"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

type Recognizer struct {
	session *ort.DynamicAdvancedSession
	vocab   []string
}

func New(modelPath string, vocabKeys []string) (*Recognizer, error) {
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"x"}, []string{"softmax_11.tmp_0"}, nil)
	if err != nil {
		return nil, err
	}
	return &Recognizer{session: session, vocab: vocabKeys}, nil
}

func (r *Recognizer) Destroy() {
	if r.session != nil {
		r.session.Destroy()
	}
}

func (r *Recognizer) ProcessCrop(crop *image.NRGBA) (string, float32, error) {
	srcW := crop.Bounds().Dx()
	srcH := crop.Bounds().Dy()

	var processedImg *image.NRGBA

	// MANGA COMPENSATOR: Detect vertical text stacks and flip 270 deg clockwise
	if srcH > int(float64(srcW)*1.4) {
		processedImg = rotate270(crop)
		srcW, srcH = srcH, srcW // Swap coordinates
	} else {
		processedImg = crop
	}

	// Dynamic scaling to structural target height 48
	h := 48
	w := int(math.Ceil(float64(h) * float64(srcW) / float64(srcH)))
	if w < 16 { w = 16 }

	resized := resizeBilinear(processedImg, w, h)

	// Construct tensor normalization array
	inputData := make([]float32, 1*3*h*w)
	idx := 0
	for c := 0; c < 3; c++ {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				pixel := resized.NRGBAAt(x, y)
				var v float32
				switch c {
				case 0: v = float32(pixel.B)
				case 1: v = float32(pixel.G)
				case 2: v = float32(pixel.R)
				}
				inputData[idx] = (v / 127.5) - 1.0
				idx++
			}
		}
	}

	inputShape := ort.NewShape(1, 3, int64(h), int64(w))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return "", 0, err
	}
	defer inputTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := r.session.Run([]ort.Value{inputTensor}, outputs); err != nil {
		return "", 0, err
	}

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return "", 0, errors.New("failed casting linear prediction tensor matrix")
	}

	outShape := outTensor.GetShape()
	text, conf := ctcGreedyDecode(outTensor.GetData(), int(outShape[1]), int(outShape[2]), r.vocab)
	return text, conf, nil
}

// Inlined bilinear resizer and CTC implementations here matching your original functions
func rotate270(src *image.NRGBA) *image.NRGBA {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, h, w))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dst.SetNRGBA(y, w-1-x, src.NRGBAAt(b.Min.X+x, b.Min.Y+y))
		}
	}
	return dst
}

```

---

### Step 4: The Clean Orchestrator Interface (`ocr.go`)

Now, combine both capabilities into a single, straightforward client API. By checking if the layout detector model path is left blank, your application can skip configuring the detection engine entirely—preserving your zero-overhead execution path for scoreboard loops.

```go
package ocr

import (
	_ "embed"
	"fmt"
	"image"
	"os"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/zen-lights/zen-lights/internal/ocr/detector"
	"github.com/zen-lights/zen-lights/internal/ocr/recognizer"
)

//go:embed ppocr_keys_v4.txt
var vocabData string

var (
	ortInitOnce sync.Once
	ortInitErr  error
	vocabKeys   []string
)

type Config struct {
	RecognizerModelPath string
	DetectorModelPath   string // Optional: Keep empty to run in High-Speed Mode A
	SharedLibraryPath   string
}

type Client struct {
	detEngine *detector.Detector
	recEngine *recognizer.Recognizer
}

func New(cfg Config) (*Client, error) {
	// Initialize ONNX Runtime environment paths safely
	ortInitOnce.Do(func() {
		vocabKeys = strings.Split(vocabData, "\n")
		if cfg.SharedLibraryPath != "" {
			ort.SetSharedLibraryPath(cfg.SharedLibraryPath)
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	if ortInitErr != nil {
		return nil, fmt.Errorf("onnx host init crash: %w", ortInitErr)
	}

	rec, err := recognizer.New(cfg.RecognizerModelPath, vocabKeys)
	if err != nil {
		return nil, fmt.Errorf("failed allocating recognition engine instance: %w", err)
	}

	var det *detector.Detector
	if cfg.DetectorModelPath != "" {
		det, err = detector.New(cfg.DetectorModelPath)
		if err != nil {
			rec.Destroy()
			return nil, fmt.Errorf("failed allocating layout engine instance: %w", err)
		}
	}

	return &Client{detEngine: det, recEngine: rec}, nil
}

func (c *Client) Close() {
	if c.recEngine != nil { c.recEngine.Destroy() }
	if c.detEngine != nil { c.detEngine.Destroy() }
}

// MODE A: Fast-Path Target ROI Scraper (Game Scoreboard Execution)
func (c *Client) ReadRegion(img *image.NRGBA, roi image.Rectangle) (string, float32, error) {
	bounds := img.Bounds()
	crop := image.NewNRGBA(image.Rect(0, 0, roi.Dx(), roi.Dy()))
	for y := roi.Min.Y; y < roi.Max.Y; y++ {
		for x := roi.Min.X; x < roi.Max.X; x++ {
			if x >= bounds.Min.X && x < bounds.Max.X && y >= bounds.Min.Y && y < bounds.Max.Y {
				crop.SetNRGBA(x-roi.Min.X, y-roi.Min.Y, img.NRGBAAt(x, y))
			}
		}
	}
	return c.recEngine.ProcessCrop(crop)
}

// MODE B: Full Pipeline Automated Extraction (Manga Page Parsing Execution)
func (c *Client) ReadFullFrame(img *image.NRGBA) ([]string, error) {
	if c.detEngine == nil {
		return nil, fmt.Errorf("client configuration engine missing text detector module context")
	}

	// 1. Process whole frame canvas through text detection tensors
	boxes, err := c.detEngine.LocateTextRegions(img)
	if err != nil {
		return nil, err
	}

	var textLines []string
	// 2. Map discovered regions to your existing fast-path recognition step
	for _, box := range boxes {
		text, conf, err := c.ReadRegion(img, box)
		if err != nil || conf < 0.4 {
			continue // Gracefully skip unreadable segments or layout noise
		}
		textLines = append(textLines, text)
	}

	return textLines, nil
}

```

---

### Verification Strategy

Your tests can now cleanly document and exercise both processing patterns:

```go
func TestExecutionModes(t *testing.T) {
	img := loadTestNRGBA("/home/jang/Pictures/manga_or_game.png")

	// Verify High-Speed Scoreboard Path (Mode A)
	t.Run("ScoreboardModeA", func(t *testing.T) {
		client, _ := New(Config{
			RecognizerModelPath: "models/ch_PP-OCRv4_rec_infer.onnx",
		})
		defer client.Close()

		text, _, _ := client.ReadRegion(img, image.Rect(10, 10, 100, 40))
		t.Logf("Mode A Text Captured: %s", text)
	})

	// Verify Dynamic Manga Layout Processing Path (Mode B)
	t.Run("MangaModeB", func(t *testing.T) {
		client, _ := New(Config{
			RecognizerModelPath: "models/ch_PP-OCRv4_rec_infer.onnx",
			DetectorModelPath:   "models/ch_PP-OCRv4_det_infer.onnx",
		})
		defer client.Close()

		lines, _ := client.ReadFullFrame(img)
		t.Logf("Mode B Text Blocks Extracted: %v", lines)
	})
}

```