package detector

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// Detector wraps the ONNX session for PP-OCRv4 DBNet text detection.
type Detector struct {
	session *ort.DynamicAdvancedSession
	thresh  float32
}

// New creates a new Detector session.
func New(modelPath string) (*Detector, error) {
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"x"},
		[]string{"sigmoid_0.tmp_0"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load detection model tensor: %w", err)
	}
	return &Detector{
		session: session,
		thresh:  0.3, // Standard DBNet threshold
	}, nil
}

// Destroy releases the underlying ONNX detection session.
func (d *Detector) Destroy() {
	if d.session != nil {
		d.session.Destroy()
		d.session = nil
	}
}

// LocateTextRegions scans an entire image and returns the identified text bounding boxes.
func (d *Detector) LocateTextRegions(img *image.NRGBA) ([]image.Rectangle, error) {
	bounds := img.Bounds()
	srcW, srcH := bounds.Dx(), bounds.Dy()
	if srcW == 0 || srcH == 0 {
		return nil, errors.New("empty source image bounds")
	}

	// 1. Calculate dynamic resizing alignment to multiples of 32 (standard DBNet restriction)
	maxWH := srcW
	if srcH > maxWH {
		maxWH = srcH
	}
	limitSideLen := 960
	if maxWH >= 1500 {
		limitSideLen = 2000
	} else if maxWH >= 960 {
		limitSideLen = 1500
	}

	ratio := 1.0
	if maxWH > limitSideLen {
		ratio = float64(limitSideLen) / float64(maxWH)
	}

	detW := int(math.Round(float64(srcW)*ratio/32) * 32)
	detH := int(math.Round(float64(srcH)*ratio/32) * 32)
	if detW < 32 {
		detW = 32
	}
	if detH < 32 {
		detH = 32
	}

	// 2. Resize original image to alignment dimensions
	resized := resizeBilinear(img, detW, detH)

	// 3. Normalize and build flat BGR channel-first array [1, 3, detH, detW]
	inputData := make([]float32, 1*3*detH*detW)
	idx := 0
	for c := 0; c < 3; c++ {
		for y := 0; y < detH; y++ {
			for x := 0; x < detW; x++ {
				pixel := resized.NRGBAAt(x, y)
				var val float32
				switch c {
				case 0:
					val = float32(pixel.B)
				case 1:
					val = float32(pixel.G)
				case 2:
					val = float32(pixel.R)
				}
				inputData[idx] = (val / 127.5) - 1.0
				idx++
			}
		}
	}

	// 4. Invoke Inference
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

	// 5. Post-processing component extraction with dilation
	return d.extractBoundingBoxes(outData, outH, outW, srcW, srcH), nil
}

// extractBoundingBoxes clusters predicted high-confidence pixels and returns dilated boxes.
func (d *Detector) extractBoundingBoxes(pred []float32, h, w, srcW, srcH int) []image.Rectangle {
	visited := make([]bool, len(pred))
	var rects []image.Rectangle

	scaleX := float64(srcW) / float64(w)
	scaleY := float64(srcH) / float64(h)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := y*w + x
			if pred[idx] > d.thresh && !visited[idx] {
				// Component detection via Breadth-First Search
				minX, maxX := x, x
				minY, maxY := y, y

				queue := []int{idx}
				visited[idx] = true

				for len(queue) > 0 {
					curr := queue[0]
					queue = queue[1:]

					cx := curr % w
					cy := curr / w

					if cx < minX {
						minX = cx
					}
					if cx > maxX {
						maxX = cx
					}
					if cy < minY {
						minY = cy
					}
					if cy > maxY {
						maxY = cy
					}

					// 4-way connected components
					neighbors := []int{
						curr - 1,
						curr + 1,
						curr - w,
						curr + w,
					}
					for _, n := range neighbors {
						if n >= 0 && n < len(pred) && !visited[n] {
							nx := n % w
							ny := n / w
							if nx >= 0 && nx < w && ny >= 0 && ny < h && pred[n] > d.thresh {
								visited[n] = true
								queue = append(queue, n)
							}
						}
					}
				}

				// Aspect scale mapping and noise filter
				wBox := maxX - minX
				hBox := maxY - minY
				if wBox > 4 && hBox > 4 {
					// Scale back to original coordinates
					minXOrig := int(float64(minX) * scaleX)
					maxXOrig := int(float64(maxX) * scaleX)
					minYOrig := int(float64(minY) * scaleY)
					maxYOrig := int(float64(maxY) * scaleY)

					// Unclip Dilation: expand box by standard 8-12% offset to prevent text clipping
					padW := int(float64(maxXOrig-minXOrig) * 0.08)
					padH := int(float64(maxYOrig-minYOrig) * 0.12)
					if padW < 2 {
						padW = 2
					}
					if padH < 2 {
						padH = 2
					}

					minXOrig -= padW
					maxXOrig += padW
					minYOrig -= padH
					maxYOrig += padH

					// Clamp back to original bounds
					if minXOrig < 0 {
						minXOrig = 0
					}
					if maxXOrig > srcW {
						maxXOrig = srcW
					}
					if minYOrig < 0 {
						minYOrig = 0
					}
					if maxYOrig > srcH {
						maxYOrig = srcH
					}

					rects = append(rects, image.Rect(minXOrig, minYOrig, maxXOrig, maxYOrig))
				}
			}
		}
	}
	return rects
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
