package svg

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/zen-lights/zen-lights/internal/paint/engine"
)

// Engine implements engine.ImageEngine for Qwen-Coder-based SVG generation.
type Engine struct {
	modelDir string
	opts     engine.Options
}

// Initialize stores the model directory and options.
func (e *Engine) Initialize(modelDir string, opts engine.Options) error {
	e.modelDir = modelDir
	e.opts = opts
	return nil
}

// Generate runs the Python SVG generator script and writes the SVG to output dir.
func (e *Engine) Generate(req engine.GenerateRequest) (engine.GenerateResult, error) {
	t0 := time.Now()

	// Ensure output directory exists
	if err := os.MkdirAll(e.opts.OutputDir, 0755); err != nil {
		return engine.GenerateResult{}, fmt.Errorf("create output dir: %w", err)
	}

	// Generate a unique output file path ending with .svg
	filename := fmt.Sprintf("zp_%d_%d.svg", time.Now().UnixNano()/1e6, req.Seed)
	outputPath := filepath.Join(e.opts.OutputDir, filename)

	// Try database search first
	icon, found := Search(req.Prompt)
	if found && isSimpleQuery(req.Prompt) {
		svgContent, err := ReadSVG(icon)
		if err == nil {
			if writeErr := os.WriteFile(outputPath, []byte(svgContent), 0644); writeErr == nil {
				elapsed := time.Since(t0)
				return engine.GenerateResult{
					ImagePath:  outputPath,
					DurationMs: elapsed.Milliseconds(),
					Width:      req.Width,
					Height:     req.Height,
					Seed:       req.Seed,
				}, nil
			}
		}
	}

	pythonBin := "/media/jang/home/Deve/torch/bin/python"
	scriptPath := "/media/jang/home/Deve/zen-lights/internal/paint/svg/generate_svg.py"
	modelID := "Qwen/Qwen2.5-Coder-1.5B-Instruct"

	args := []string{scriptPath, req.Prompt, outputPath, modelID}
	if found {
		args = append(args, filepath.Join(baseDataPath, icon.Dataset, icon.Filename))
	}

	cmd := exec.Command(pythonBin, args...)
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	err := cmd.Run()
	if err != nil {
		return engine.GenerateResult{}, fmt.Errorf("execute generate_svg.py: %w, stderr: %s", err, stderrBuf.String())
	}

	// Verify that the output SVG file exists
	if _, err := os.Stat(outputPath); os.IsNotExist(err) {
		return engine.GenerateResult{}, fmt.Errorf("expected SVG file was not created at %s", outputPath)
	}

	elapsed := time.Since(t0)

	return engine.GenerateResult{
		ImagePath:  outputPath,
		DurationMs: elapsed.Milliseconds(),
		Width:      req.Width,
		Height:     req.Height,
		Seed:       req.Seed,
	}, nil
}

// Info returns a human-readable description of the engine.
func (e *Engine) Info() string {
	return fmt.Sprintf("Qwen2.5-Coder SVG engine | dir=%s", e.modelDir)
}

// Close releases any resources (no-op for SVG engine).
func (e *Engine) Close() error {
	return nil
}

func isSimpleQuery(prompt string) bool {
	cleaned := cleanQuery(prompt)
	words := strings.Split(cleaned, "-")
	return len(words) <= 2
}
