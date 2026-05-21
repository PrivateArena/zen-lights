package recognizer

import (
	"errors"
	"image"
	"image/color"
	"math"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// Recognizer wraps the ONNX session for PP-OCRv4 text line recognition.
type Recognizer struct {
	session *ort.DynamicAdvancedSession
	vocab   []string
}

// New creates a new Recognizer session.
func New(modelPath string, vocabKeys []string) (*Recognizer, error) {
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"x"},
		[]string{"softmax_11.tmp_0"},
		nil,
	)
	if err != nil {
		return nil, err
	}
	return &Recognizer{
		session: session,
		vocab:   vocabKeys,
	}, nil
}

// Destroy releases the underlying ONNX recognition session.
func (r *Recognizer) Destroy() {
	if r.session != nil {
		r.session.Destroy()
		r.session = nil
	}
}

// ProcessCrop processes a cropped text region, normalising it and decoding it.
func (r *Recognizer) ProcessCrop(crop *image.NRGBA) (string, float32, error) {
	srcW := crop.Bounds().Dx()
	srcH := crop.Bounds().Dy()
	if srcW == 0 || srcH == 0 {
		return "", 0, errors.New("empty crop bounds")
	}

	var processedImg *image.NRGBA
	// Manga vertical stack compensator: rotate CJK vertical layout 270 deg clockwise (90 counter-clockwise)
	if srcH > int(float64(srcW)*1.4) {
		processedImg = rotate270(crop)
		srcW, srcH = srcH, srcW // swap dimensions
	} else {
		processedImg = crop
	}

	// Rescale to height 48, maintaining aspect ratio
	h := 48
	w := int(math.Ceil(float64(h) * float64(srcW) / float64(srcH)))
	if w < 16 {
		w = 16
	}

	resized := resizeBilinear(processedImg, w, h)

	// Flat BGR normalization input [1, 3, 48, w]
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

	inputShape := ort.NewShape(1, 3, int64(h), int64(w))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return "", 0, err
	}
	defer inputTensor.Destroy()

	inputs := []ort.Value{inputTensor}
	outputs := []ort.Value{nil}

	err = r.session.Run(inputs, outputs)
	if err != nil {
		return "", 0, err
	}

	outTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return "", 0, errors.New("failed casting output to float32 tensor")
	}

	outShape := outTensor.GetShape()
	seqLen := int(outShape[1])
	numClasses := int(outShape[2])

	text, conf := ctcGreedyDecode(outTensor.GetData(), seqLen, numClasses, r.vocab)
	return text, conf, nil
}

// rotate270 rotates an image 270 degrees clockwise (90 degrees counter-clockwise).
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

// resizeBilinear resizes an image using bilinear interpolation.
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
