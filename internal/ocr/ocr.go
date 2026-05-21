// Package ocr wraps PaddleOCR/RapidOCR PP-OCRv4 running on ONNX Runtime
// and exposes a single method optimised for reading text/digits from cropped regions.
package ocr

import (
	_ "embed"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/zen-lights/zen-lights/internal/imgutil"
	"github.com/zen-lights/zen-lights/pkg/game"
)

//go:embed ppocr_keys_v4.txt
var vocabData string

var (
	ortInitOnce sync.Once
	ortInitErr  error
	vocabKeys   []string
)

// Options configures OCR preprocessing.
type Options struct {
	scaleFactor       int
	thresholdValue    uint8
	invertThreshold   bool
	adaptiveThreshold bool
}

// DefaultOptions returns sensible defaults for bright-text-on-dark MOBAs.
func DefaultOptions() Options {
	return Options{
		scaleFactor:    3,
		thresholdValue: 110,
	}
}

// OptionsFromConfig builds Options from a game.OCRConfig.
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

// Client holds a long-lived ONNX Runtime session configured for text recognition.
type Client struct {
	session *ort.DynamicAdvancedSession
	opts    Options
}

// initORT sets the shared library path and initializes the environment.
func initORT() error {
	ortInitOnce.Do(func() {
		// Parse vocabulary
		vocabKeys = strings.Split(vocabData, "\n")

		// Resolve library path
		libPath := os.Getenv("ORT_SHARED_LIB_PATH")
		if libPath == "" {
			candidates := []string{
				"/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2",
				"./piper/libonnxruntime.so.1.24.2",
				"../piper/libonnxruntime.so.1.24.2",
				"../../piper/libonnxruntime.so.1.24.2",
				"./models/libonnxruntime.so",
			}
			for _, c := range candidates {
				abs, err := filepath.Abs(c)
				if err == nil {
					if _, err := os.Stat(abs); err == nil {
						libPath = abs
						break
					}
				}
			}
		}
		if libPath == "" {
			libPath = "libonnxruntime.so" // default system loader fallback
		}

		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// New creates a new OCR client powered by ONNX Runtime.
func New(opts Options) (*Client, error) {
	if err := initORT(); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}

	// Resolve model path
	modelPath := os.Getenv("PPOCR_MODEL_PATH")
	if modelPath == "" {
		candidates := []string{
			"/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_rec_infer.onnx",
			"./models/ch_PP-OCRv4_rec_infer.onnx",
			"../models/ch_PP-OCRv4_rec_infer.onnx",
			"../../models/ch_PP-OCRv4_rec_infer.onnx",
		}
		for _, c := range candidates {
			abs, err := filepath.Abs(c)
			if err == nil {
				if _, err := os.Stat(abs); err == nil {
					modelPath = abs
					break
				}
			}
		}
	}
	if modelPath == "" {
		modelPath = "models/ch_PP-OCRv4_rec_infer.onnx"
	}

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"x"},
		[]string{"softmax_11.tmp_0"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create advanced session: %w", err)
	}

	return &Client{
		session: session,
		opts:    opts,
	}, nil
}

// Close releases the underlying ONNX Runtime session. Always defer this.
func (c *Client) Close() {
	if c.session != nil {
		c.session.Destroy()
		c.session = nil
	}
}

// ReadRegion crops roi from the raw RGB24 frame, preprocesses it, and returns
// the recognised text.
func (c *Client) ReadRegion(frameData []byte, frameW, frameH int, roi image.Rectangle) (string, error) {
	text, _, err := c.ReadRegionWithDiagnostics(frameData, frameW, frameH, roi, false)
	return text, err
}

// ReadRegionWithDiagnostics is like ReadRegion but optionally also returns the
// preprocessed PNG for debugging.
func (c *Client) ReadRegionWithDiagnostics(
	frameData []byte, frameW, frameH int, roi image.Rectangle, dumpPNG bool,
) (text string, debugPNG []byte, err error) {
	// 1. Crop
	crop := imgutil.CropRGB24(frameData, frameW, frameH, roi)

	// 2. Preprocess: Resize to height 48 and keep aspect ratio
	h := 48
	w := int(math.Ceil(float64(h) * float64(roi.Dx()) / float64(roi.Dy())))
	if w < 16 {
		w = 16
	}
	resized := resizeBilinear(crop, w, h)

	// 3. Optional debug PNG
	if dumpPNG {
		pngBytes, encErr := imgutil.EncodePNG(resized)
		if encErr == nil {
			debugPNG = pngBytes
		}
	}

	// 4. Build flat BGR normalised input array [1, 3, 48, w]
	inputData := make([]float32, 1*3*h*w)
	idx := 0
	// Blue
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			inputData[idx] = float32(resized.NRGBAAt(x, y).B)/127.5 - 1.0
			idx++
		}
	}
	// Green
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			inputData[idx] = float32(resized.NRGBAAt(x, y).G)/127.5 - 1.0
			idx++
		}
	}
	// Red
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
		return "", debugPNG, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{nil}

	// Run Inference
	err = c.session.Run(inputs, outputs)
	if err != nil {
		return "", debugPNG, fmt.Errorf("session run: %w", err)
	}

	// Process output
	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return "", debugPNG, fmt.Errorf("cast output tensor failed")
	}

	outData := outTensor.GetData()
	outShape := outTensor.GetShape()

	seqLen := int(outShape[1])
	numClasses := int(outShape[2])

	// Decode using CTC greedy decoding
	decodedText, _ := ctcGreedyDecode(outData, seqLen, numClasses, vocabKeys)
	return decodedText, debugPNG, nil
}

// resizeBilinear resizes an image to the given width and height using bilinear interpolation.
func resizeBilinear(img *image.NRGBA, w, h int) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			gx := float64(x) / float64(w) * float64(srcW)
			gy := float64(y) / float64(h) * float64(srcH)
			gxi := int(gx)
			gyi := int(gy)

			x0 := gxi
			if x0 > srcW-1 {
				x0 = srcW - 1
			}
			x1 := gxi + 1
			if x1 > srcW-1 {
				x1 = srcW - 1
			}
			y0 := gyi
			if y0 > srcH-1 {
				y0 = srcH - 1
			}
			y1 := gyi + 1
			if y1 > srcH-1 {
				y1 = srcH - 1
			}

			dx := gx - float64(gxi)
			dy := gy - float64(gyi)

			c00 := img.NRGBAAt(img.Bounds().Min.X+x0, img.Bounds().Min.Y+y0)
			c10 := img.NRGBAAt(img.Bounds().Min.X+x1, img.Bounds().Min.Y+y0)
			c01 := img.NRGBAAt(img.Bounds().Min.X+x0, img.Bounds().Min.Y+y1)
			c11 := img.NRGBAAt(img.Bounds().Min.X+x1, img.Bounds().Min.Y+y1)

			r := float64(c00.R)*(1-dx)*(1-dy) + float64(c10.R)*dx*(1-dy) + float64(c01.R)*(1-dx)*dy + float64(c11.R)*dx*dy
			g := float64(c00.G)*(1-dx)*(1-dy) + float64(c10.G)*dx*(1-dy) + float64(c01.G)*(1-dx)*dy + float64(c11.G)*dx*dy
			b := float64(c00.B)*(1-dx)*(1-dy) + float64(c10.B)*dx*(1-dy) + float64(c01.B)*(1-dx)*dy + float64(c11.B)*dx*dy

			out.SetNRGBA(x, y, color.NRGBA{R: uint8(r), G: uint8(g), B: uint8(b), A: 255})
		}
	}
	return out
}

// ctcGreedyDecode decodes prediction logits using CTC greedy search.
func ctcGreedyDecode(preds []float32, seqLen, numClasses int, keys []string) (string, float32) {
	var sb strings.Builder
	lastIdx := -1
	var confSum float32
	var confCount int

	for t := 0; t < seqLen; t++ {
		row := preds[t*numClasses : (t+1)*numClasses]

		// Find argmax
		maxIdx := 0
		maxVal := row[0]
		for i := 1; i < numClasses; i++ {
			if row[i] > maxVal {
				maxVal = row[i]
				maxIdx = i
			}
		}

		// Filter blank (0) and duplicates
		if maxIdx != 0 && maxIdx != lastIdx {
			if maxIdx < len(keys) {
				sb.WriteString(keys[maxIdx])
				confSum += maxVal
				confCount++
			}
		}
		lastIdx = maxIdx
	}

	confidence := float32(0.0)
	if confCount > 0 {
		confidence = confSum / float32(confCount)
	}
	return sb.String(), confidence
}
