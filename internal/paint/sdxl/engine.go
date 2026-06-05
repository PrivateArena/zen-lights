// Package sdxl implements an ONNX-based SDXL/SDXL-Turbo/LCM image generation pipeline.
//
// Pipeline stages:
//  1. Text Encoder (CLIP ViT-L/14) → token embeddings + pooled output
//  2. UNet (conditioned denoising) → latent diffusion
//  3. Scheduler (LCM / DDIM in pure Go)
//  4. VAE Decoder → RGB pixels
//  5. PNG encode → disk
//
// Expected model directory layout:
//
//	models/<name>/
//	  model.json
//	  text_encoder.onnx
//	  unet.onnx
//	  vae_decoder.onnx
//	  tokenizer/vocab.json   (optional — falls back to simple whitespace tokenizer)
package sdxl

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	ort "github.com/yalue/onnxruntime_go"
	zenort "github.com/zen-lights/zen-lights/internal/paint/ort"
	"github.com/zen-lights/zen-lights/internal/paint/tokenizer"

	"github.com/zen-lights/zen-lights/internal/paint/engine"
)

// Engine is the SDXL/SDXL-Turbo/LCM backend.
type Engine struct {
	textEncoder        *ort.DynamicAdvancedSession
	unet               *ort.DynamicAdvancedSession
	vaeDecoder         *ort.DynamicAdvancedSession
	tokenizer          *tokenizer.ClipTokenizer
	opts               engine.Options
	modelDir           string
	info               string
	unetInputs         []string
	textEncoderOutputs []string
	timestepIsFloat    bool
	alphasCumprod      [1000]float32
}

// Initialize loads ONNX sessions from modelDir.
func (e *Engine) Initialize(modelDir string, opts engine.Options) error {
	if err := zenort.Init(opts.OrtLibPath()); err != nil {
		return fmt.Errorf("ort init: %w", err)
	}

	e.modelDir = modelDir
	e.opts = opts

	threads := opts.NumThreads
	if threads <= 0 {
		threads = runtime.NumCPU()
		if threads > 16 {
			threads = 16
		}
	}

	sessOpts, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("session options: %w", err)
	}
	_ = sessOpts.SetIntraOpNumThreads(threads)
	_ = sessOpts.SetInterOpNumThreads(1)

	switch strings.ToLower(opts.ExecutionProvider) {
	case "rocm", "hip":
		err = sessOpts.AppendExecutionProvider("ROCM", map[string]string{
			"device_id": "0",
		})
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append ROCm provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled ROCm GPU execution provider")
		}
	case "cuda":
		cudaOpts, cudaErr := ort.NewCUDAProviderOptions()
		if cudaErr == nil {
			defer cudaOpts.Destroy()
			cudaErr = sessOpts.AppendExecutionProviderCUDA(cudaOpts)
		}
		if cudaErr != nil {
			fmt.Printf("[ort] Warning: failed to append CUDA provider: %v. Using CPU fallback.\n", cudaErr)
		} else {
			fmt.Println("[ort] Enabled CUDA GPU execution provider")
		}
	case "directml":
		err = sessOpts.AppendExecutionProviderDirectML(0)
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append DirectML provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled DirectML GPU execution provider")
		}
	case "openvino":
		err = sessOpts.AppendExecutionProviderOpenVINO(nil)
		if err != nil {
			fmt.Printf("[ort] Warning: failed to append OpenVINO provider: %v. Using CPU fallback.\n", err)
		} else {
			fmt.Println("[ort] Enabled OpenVINO execution provider")
		}
	}

	load := func(name string) (*ort.DynamicAdvancedSession, []string, []string, error) {
		path := filepath.Join(modelDir, name)
		if _, err := os.Stat(path); err != nil {
			// Try subdirectory fallback (e.g. text_encoder.onnx -> text_encoder/model.onnx)
			base := strings.TrimSuffix(name, ".onnx")
			fallbackPath := filepath.Join(modelDir, base, "model.onnx")
			if _, errSub := os.Stat(fallbackPath); errSub == nil {
				path = fallbackPath
			} else {
				// Try base/name (e.g. text_encoder/text_encoder.onnx)
				fallbackPath2 := filepath.Join(modelDir, base, name)
				if _, errSub2 := os.Stat(fallbackPath2); errSub2 == nil {
					path = fallbackPath2
				} else {
					return nil, nil, nil, fmt.Errorf("model file not found: %s", path)
				}
			}
		}
		ins, outs, err := ort.GetInputOutputInfo(path)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("get io info %s: %w", name, err)
		}
		inNames := make([]string, len(ins))
		for i, v := range ins {
			inNames[i] = v.Name
			if name == "unet.onnx" && v.Name == "timestep" {
				if strings.Contains(strings.ToLower(v.DataType.String()), "float") {
					e.timestepIsFloat = true
				}
			}
			fmt.Printf("[%s input %d] Name: %s, DataType: %s, Dimensions: %v\n", name, i, v.Name, v.DataType.String(), v.Dimensions)
		}
		outNames := make([]string, len(outs))
		for i, v := range outs {
			outNames[i] = v.Name
			fmt.Printf("[%s output %d] Name: %s, DataType: %s, Dimensions: %v\n", name, i, v.Name, v.DataType.String(), v.Dimensions)
		}
		sess, err := ort.NewDynamicAdvancedSession(path, inNames, outNames, sessOpts)
		return sess, inNames, outNames, err
	}

	var textEncoderOutputs, unetInputs []string
	if e.textEncoder, _, textEncoderOutputs, err = load("text_encoder.onnx"); err != nil {
		return err
	}
	e.textEncoderOutputs = textEncoderOutputs

	if e.unet, unetInputs, _, err = load("unet.onnx"); err != nil {
		return err
	}
	e.unetInputs = unetInputs

	if e.vaeDecoder, _, _, err = load("vae_decoder.onnx"); err != nil {
		return err
	}

	e.tokenizer, err = tokenizer.NewClipTokenizer(modelDir)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}

	e.initScheduler()

	e.info = fmt.Sprintf("SDXL pipeline | dir=%s threads=%d provider=%s",
		modelDir, threads, opts.ExecutionProvider)
	return nil
}

// Generate runs the full diffusion pipeline for a single request.
func (e *Engine) Generate(req engine.GenerateRequest) (engine.GenerateResult, error) {
	t0 := time.Now()

	rng := rand.New(rand.NewSource(req.Seed))

	// 1. Tokenize + encode text
	condEmbeds, pooled, err := e.encodeText(req.Prompt)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("text encode: %w", err)
	}
	uncondEmbeds, uncondPooled, err := e.encodeText(req.NegativePrompt)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("text encode (neg): %w", err)
	}

	// 2. Init latent noise — SDXL latent space is H/8 × W/8 × 4 channels
	latentH := req.Height / 8
	latentW := req.Width / 8
	latents := randFloat32(rng, 1, 4, latentH, latentW)

	// 3. Scheduler: LCM timesteps
	timesteps := lcmTimesteps(req.Steps)

	// 4. Denoising loop
	for stepIdx := range timesteps {
		latents, err = e.denoisingStep(rng, latents, condEmbeds, uncondEmbeds, pooled, uncondPooled,
			stepIdx, timesteps, req.CFGScale, req.Width, req.Height)
		if err != nil {
			return engine.GenerateResult{}, fmt.Errorf("denoising step idx=%d: %w", stepIdx, err)
		}
	}

	// 5. Decode latents → pixels
	pixels, err := e.decodeLatents(latents)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("vae decode: %w", err)
	}

	// 6. Save PNG
	imgPath, err := saveImage(pixels, req.Width, req.Height, e.opts.OutputDir, req.Seed)
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("save png: %w", err)
	}

	return engine.GenerateResult{
		ImagePath:  imgPath,
		DurationMs: time.Since(t0).Milliseconds(),
		Width:      req.Width,
		Height:     req.Height,
		Seed:       req.Seed,
	}, nil
}

// Info returns a human-readable description of the loaded pipeline.
func (e *Engine) Info() string { return e.info }

// Close releases all ONNX sessions.
func (e *Engine) Close() error {
	if e.textEncoder != nil {
		_ = e.textEncoder.Destroy()
	}
	if e.unet != nil {
		_ = e.unet.Destroy()
	}
	if e.vaeDecoder != nil {
		_ = e.vaeDecoder.Destroy()
	}
	return nil
}

// --- Internal pipeline steps ---

// encodeText tokenizes and encodes a prompt into CLIP embeddings.
// Returns (sequence_embeddings [1, 77, 768], pooled [1, 768]).
// Uses a placeholder identity tokenizer — swap with BPE for production.
func (e *Engine) encodeText(prompt string) ([]float32, []float32, error) {
	tokens := e.tokenizer.Encode(prompt, 77)
	if len(tokens) >= 15 {
		fmt.Printf("[paint] encodeText prompt=%q tokens=%v\n", prompt, tokens[:15])
	} else {
		fmt.Printf("[paint] encodeText prompt=%q tokens=%v\n", prompt, tokens)
	}
	tokenTensor, err := ort.NewTensor(ort.NewShape(1, 77), tokens)
	if err != nil {
		return nil, nil, err
	}
	defer tokenTensor.Destroy()

	outSeq := make([]float32, 1*77*768)
	seqTensor, _ := ort.NewTensor(ort.NewShape(1, 77, 768), outSeq)
	defer seqTensor.Destroy()

	var pool []float32
	if len(e.textEncoderOutputs) >= 2 {
		outPool := make([]float32, 1*768)
		poolTensor, _ := ort.NewTensor(ort.NewShape(1, 768), outPool)
		defer poolTensor.Destroy()
		err = e.textEncoder.Run([]ort.Value{tokenTensor}, []ort.Value{seqTensor, poolTensor})
		if err == nil {
			pool = make([]float32, len(outPool))
			copy(pool, poolTensor.GetData())
		}
	} else {
		err = e.textEncoder.Run([]ort.Value{tokenTensor}, []ort.Value{seqTensor})
	}
	if err != nil {
		return nil, nil, err
	}

	seq := make([]float32, len(outSeq))
	copy(seq, seqTensor.GetData())
	return seq, pool, nil
}

// denoisingStep runs one UNet forward pass and returns updated latents.
func (e *Engine) denoisingStep(
	rng *rand.Rand,
	latents, condEmb, uncondEmb, condPool, uncondPool []float32,
	stepIdx int, timesteps []int, cfg float32,
	width, height int,
) ([]float32, error) {
	t := timesteps[stepIdx]
	latentH := height / 8
	latentW := width / 8
	latentSize := 1 * 4 * latentH * latentW

	// Classifier-free guidance: batch=[uncond, cond]
	batchLatents := make([]float32, 2*latentSize)
	copy(batchLatents[:latentSize], latents)
	copy(batchLatents[latentSize:], latents)

	batchEmbeds := append(uncondEmb, condEmb...)
	batchPool := append(uncondPool, condPool...)
	latTensor, _ := ort.NewTensor(ort.NewShape(2, 4, int64(latentH), int64(latentW)), batchLatents)
	embTensor, _ := ort.NewTensor(ort.NewShape(2, 77, 768), batchEmbeds)
	defer latTensor.Destroy()
	defer embTensor.Destroy()

	var tTensor ort.Value
	if e.timestepIsFloat {
		tTensor, _ = ort.NewTensor(ort.NewShape(2), []float32{float32(t), float32(t)})
	} else {
		tTensor, _ = ort.NewTensor(ort.NewShape(2), []int64{int64(t), int64(t)})
	}
	defer tTensor.Destroy()

	var inputs []ort.Value
	if len(e.unetInputs) == 3 {
		inputs = []ort.Value{latTensor, tTensor, embTensor}
	} else if len(e.unetInputs) == 4 {
		fourthInputName := e.unetInputs[3]
		if fourthInputName == "timestep_cond" {
			// LCM style: expects [batch, 256]
			timestepCondData := make([]float32, 2*256)
			timestepCondTensor, _ := ort.NewTensor(ort.NewShape(2, 256), timestepCondData)
			defer timestepCondTensor.Destroy()
			inputs = []ort.Value{latTensor, tTensor, embTensor, timestepCondTensor}
		} else {
			// SDXL style: expects [batch, 768] (poolTensor)
			poolTensor, _ := ort.NewTensor(ort.NewShape(2, 768), batchPool)
			defer poolTensor.Destroy()
			inputs = []ort.Value{latTensor, tTensor, embTensor, poolTensor}
		}
	} else {
		inputs = []ort.Value{latTensor, tTensor, embTensor}
	}

	outNoise := make([]float32, 2*latentSize)
	noiseTensor, _ := ort.NewTensor(ort.NewShape(2, 4, int64(latentH), int64(latentW)), outNoise)
	defer noiseTensor.Destroy()

	if err := e.unet.Run(inputs, []ort.Value{noiseTensor}); err != nil {
		return nil, err
	}

	noiseData := noiseTensor.GetData()
	uncondNoise := noiseData[:latentSize]
	condNoise := noiseData[latentSize:]

	// CFG: noise = uncond + cfg_scale * (cond - uncond)
	guided := make([]float32, latentSize)
	for i := range guided {
		guided[i] = uncondNoise[i] + cfg*(condNoise[i]-uncondNoise[i])
	}

	// LCM scheduler update step
	alphaProdT := e.alphasCumprod[t]
	betaProdT := 1.0 - alphaProdT
	sqrtAlphaProdT := float32(math.Sqrt(float64(alphaProdT)))
	sqrtBetaProdT := float32(math.Sqrt(float64(betaProdT)))

	// 1. Estimate x0 (predOriginalSample)
	predX0 := make([]float32, latentSize)
	for i := 0; i < latentSize; i++ {
		predX0[i] = (latents[i] - sqrtBetaProdT*guided[i]) / sqrtAlphaProdT
	}

	// 2. Get boundary condition scalings c_skip and c_out
	scaledTimestep := float64(t) * 10.0
	denom := scaledTimestep*scaledTimestep + 0.25
	cSkip := float32(0.25 / denom)
	cOut := float32(scaledTimestep / math.Sqrt(denom))

	// 3. Compute denoised latent
	denoised := make([]float32, latentSize)
	for i := 0; i < latentSize; i++ {
		denoised[i] = cSkip*latents[i] + cOut*predX0[i]
	}

	// 4. If not last step, step to tPrev
	if stepIdx < len(timesteps)-1 {
		tPrev := timesteps[stepIdx+1]
		alphaProdTPrev := e.alphasCumprod[tPrev]
		sqrtAlphaProdTPrev := float32(math.Sqrt(float64(alphaProdTPrev)))
		sqrtBetaProdTPrev := float32(math.Sqrt(float64(1.0 - alphaProdTPrev)))

		// Sample Gaussian noise
		noise := make([]float32, latentSize)
		for i := 0; i < latentSize; i++ {
			noise[i] = float32(rng.NormFloat64())
		}

		// LCM prev_sample step formula
		updated := make([]float32, latentSize)
		for i := 0; i < latentSize; i++ {
			updated[i] = sqrtAlphaProdTPrev*denoised[i] + sqrtBetaProdTPrev*noise[i]
		}
		return updated, nil
	}

	return denoised, nil
}

// decodeLatents runs the VAE decoder on the final latent tensor.
func (e *Engine) decodeLatents(latents []float32) ([]float32, error) {
	// Undo VAE scaling factor
	for i := range latents {
		latents[i] /= 0.18215
	}

	// Shape inferred from latent size — assume square if ambiguous
	n := len(latents) / 4
	side := int(math.Round(math.Sqrt(float64(n))))
	inTensor, _ := ort.NewTensor(ort.NewShape(1, 4, int64(side), int64(side)), latents)
	defer inTensor.Destroy()

	outPixels := make([]float32, 1*3*side*8*side*8)
	outTensor, _ := ort.NewTensor(ort.NewShape(1, 3, int64(side*8), int64(side*8)), outPixels)
	defer outTensor.Destroy()

	if err := e.vaeDecoder.Run([]ort.Value{inTensor}, []ort.Value{outTensor}); err != nil {
		return nil, err
	}

	result := make([]float32, len(outPixels))
	copy(result, outTensor.GetData())
	return result, nil
}

// --- Scheduler helpers ---

// lcmTimesteps returns LCM-style timesteps in descending order.
func lcmTimesteps(steps int) []int {
	if steps <= 0 {
		steps = 4
	}
	// Standard LCM schedule: evenly spaced in [999, 0]
	ts := make([]int, steps)
	if steps == 1 {
		ts[0] = 999
		return ts
	}
	for i := 0; i < steps; i++ {
		ts[i] = 999 - i*(999/(steps-1))
		if ts[i] < 0 {
			ts[i] = 0
		}
	}
	return ts
}

// lcmSigma returns the noise scale for a given timestep.
func lcmSigma(t int) float64 {
	// Simplified linear sigma schedule
	return float64(t) / 1000.0
}

// --- Tokenizer ---

// simpleTokenize converts a prompt to a padded int32 token slice.
// This is a placeholder — replace with actual BPE tokenizer for real use.
func simpleTokenize(prompt string, maxLen int) []int32 {
	tokens := make([]int32, maxLen)
	tokens[0] = 49406 // <|startoftext|>
	for i, b := range []byte(prompt) {
		if i+1 >= maxLen-1 {
			break
		}
		tokens[i+1] = int32(b)
	}
	tokens[maxLen-1] = 49407 // <|endoftext|>
	return tokens
}

// --- Tensor / image utilities ---

// randFloat32 fills a float32 slice with standard normal noise.
func randFloat32(rng *rand.Rand, dims ...int) []float32 {
	size := 1
	for _, d := range dims {
		size *= d
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	return data
}

// saveImage converts CHW float32 pixels in [-1,1] to an RGBA PNG on disk.
func saveImage(pixels []float32, width, height int, outputDir string, seed int64) (string, error) {
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	pixPerCh := width * height
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x
			r := clampUint8(pixels[0*pixPerCh+idx])
			g := clampUint8(pixels[1*pixPerCh+idx])
			b := clampUint8(pixels[2*pixPerCh+idx])
			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("zp_%d_%d.png", time.Now().UnixMilli(), seed)
	path := filepath.Join(outputDir, name)
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return path, png.Encode(f, img)
}

// clampUint8 converts a float in [-1, 1] to uint8 [0, 255].
func clampUint8(v float32) uint8 {
	v = (v + 1.0) / 2.0
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return uint8(v * 255)
}

func (e *Engine) initScheduler() {
	betaStart := 0.00085
	betaEnd := 0.0120
	start := math.Sqrt(betaStart)
	end := math.Sqrt(betaEnd)

	betas := make([]float64, 1000)
	for i := 0; i < 1000; i++ {
		val := start + float64(i)*(end-start)/999.0
		betas[i] = val * val
	}

	product := 1.0
	for i := 0; i < 1000; i++ {
		alpha := 1.0 - betas[i]
		product *= alpha
		e.alphasCumprod[i] = float32(product)
	}
}
