// Package ocr wraps PaddleOCR/RapidOCR PP-OCRv4 running on ONNX Runtime
// and exposes simple client APIs optimized for structured scoreboard ROI scraping (Mode A)
// and full automated page layout parsing (Mode B).
package ocr

import (
	_ "embed"
	"fmt"
	"image"
	"image/color"
	"os"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
	"github.com/zen-lights/zen-lights/internal/imgutil"
	"github.com/zen-lights/zen-lights/internal/ocr/detector"
	"github.com/zen-lights/zen-lights/internal/ocr/recognizer"
	"github.com/zen-lights/zen-lights/pkg/game"
)

//go:embed ppocr_keys_v4.txt
var vocabData string

var (
	ortInitOnce sync.Once
	ortInitErr  error
	vocabKeys   []string
)

// Options configures OCR assets and preprocessing.
type Options struct {
	RecModelPath      string
	RecVocabPath      string
	DetModelPath      string
	scaleFactor       int
	thresholdValue    uint8
	invertThreshold   bool
	adaptiveThreshold bool
}

// DefaultOptions returns sensible defaults for bright-text-on-dark MOBAs.
func DefaultOptions() Options {
	return Options{
		RecModelPath:   "/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_rec_infer.onnx",
		DetModelPath:   "/media/jang/home/Deve/zen-lights/models/ch_PP-OCRv4_det_infer.onnx",
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

// TextResult represents a single recognized text block with its layout coordinates.
type TextResult struct {
	Text       string
	Confidence float32
	Bounds     image.Rectangle
}

// Client holds long-lived ONNX Runtime recognition and detection sessions.
type Client struct {
	recEngine *recognizer.Recognizer
	detEngine *detector.Detector
	opts      Options
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

	activeVocab := vocabKeys
	if opts.RecVocabPath != "" {
		data, err := os.ReadFile(opts.RecVocabPath)
		if err != nil {
			return nil, fmt.Errorf("read external vocab: %w", err)
		}
		// Split and clean up lines (remove CR if present on Windows-origin files)
		activeVocab = strings.Split(strings.ReplaceAll(string(data), "\r", ""), "\n")
	}

	// Resolve recognizer model path
	recModelPath := opts.RecModelPath
	if recModelPath == "" {
		recModelPath = os.Getenv("PPOCR_MODEL_PATH")
	}
	if recModelPath == "" {
		recModelPath = "models/ch_PP-OCRv4_rec_infer.onnx"
	}

	recEngine, err := recognizer.New(recModelPath, activeVocab)
	if err != nil {
		return nil, fmt.Errorf("create recognizer session: %w", err)
	}

	// Resolve optional detector model path
	detModelPath := opts.DetModelPath
	if detModelPath == "" {
		detModelPath = os.Getenv("PPOCR_DET_MODEL_PATH")
	}

	var detEngine *detector.Detector
	if detModelPath != "" {
		var detErr error
		detEngine, detErr = detector.New(detModelPath)
		if detErr != nil {
			recEngine.Destroy()
			return nil, fmt.Errorf("create detector session: %w", detErr)
		}
	}

	return &Client{
		recEngine: recEngine,
		detEngine: detEngine,
		opts:      opts,
	}, nil
}

// Close releases the underlying ONNX Runtime sessions. Always defer this.
func (c *Client) Close() {
	if c.recEngine != nil {
		c.recEngine.Destroy()
		c.recEngine = nil
	}
	if c.detEngine != nil {
		c.detEngine.Destroy()
		c.detEngine = nil
	}
}

// ReadRegion crops roi from the raw RGB24 frame, preprocesses it, and returns the recognised text.
func (c *Client) ReadRegion(frameData []byte, frameW, frameH int, roi image.Rectangle) (string, error) {
	text, _, err := c.ReadRegionWithDiagnostics(frameData, frameW, frameH, roi, false)
	return text, err
}

// ReadRegionWithDiagnostics is like ReadRegion but optionally also returns the preprocessed PNG for debugging.
func (c *Client) ReadRegionWithDiagnostics(
	frameData []byte, frameW, frameH int, roi image.Rectangle, dumpPNG bool,
) (text string, debugPNG []byte, err error) {
	// 1. Crop
	crop := imgutil.CropRGB24(frameData, frameW, frameH, roi)

	// 2. Optional debug PNG
	if dumpPNG {
		pngBytes, encErr := imgutil.EncodePNG(crop)
		if encErr == nil {
			debugPNG = pngBytes
		}
	}

	// 3. Process recognition
	text, _, err = c.recEngine.ProcessCrop(crop)
	return text, debugPNG, err
}

// ReadFullFrame scans an entire frame buffer, automatically segments all text layouts,
// and processes them sequentially to extract text lines.
func (c *Client) ReadFullFrame(frameData []byte, frameW, frameH int) ([]TextResult, error) {
	if c.detEngine == nil {
		return nil, fmt.Errorf("layout detector engine is not initialized (PPOCR_DET_MODEL_PATH not set or model file missing)")
	}

	// 1. Create NRGBA image from raw frameData
	img := image.NewNRGBA(image.Rect(0, 0, frameW, frameH))
	for y := 0; y < frameH; y++ {
		srcRow := frameData[y*frameW*3:]
		for x := 0; x < frameW; x++ {
			i := x * 3
			img.SetNRGBA(x, y, color.NRGBA{
				R: srcRow[i],
				G: srcRow[i+1],
				B: srcRow[i+2],
				A: 255,
			})
		}
	}

	// 2. Locate all text regions
	boxes, err := c.detEngine.LocateTextRegions(img)
	if err != nil {
		return nil, fmt.Errorf("text detection failed: %w", err)
	}

	// 3. Sequential recognition on all regions
	var results []TextResult
	for _, box := range boxes {
		// Crop the detected box
		crop := imgutil.CropRGB24(frameData, frameW, frameH, box)
		text, conf, err := c.recEngine.ProcessCrop(crop)
		if err != nil || conf < 0.4 {
			continue // skip layout noise or unreadable regions
		}

		results = append(results, TextResult{
			Text:       text,
			Confidence: conf,
			Bounds:     box,
		})
	}

	return results, nil
}
