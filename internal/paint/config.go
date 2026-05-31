package paint

import (
	"fmt"
)

// Config holds the paint engine configuration loaded from the main config.json.
type Config struct {
	DefaultModel      string `json:"default_model"`
	ExecutionProvider string `json:"execution_provider"`
	NumThreads        int    `json:"num_threads"`
	OutputDir         string `json:"output_dir"`
	OrtLibPath        string `json:"ort_lib_path"`
	MaxConcurrency    int    `json:"max_concurrency"`
	ModelsDir         string `json:"models_dir"`
}

// DefaultConfig returns the default fallback configurations.
var DefaultConfig = Config{
	DefaultModel:      "sdxl-turbo",
	ExecutionProvider: "cpu",
	NumThreads:        0,
	OutputDir:         "/tmp/zen-paint",
	OrtLibPath:        "",
	MaxConcurrency:    1,
	ModelsDir:         "./models",
}

// ModelDir returns the absolute or relative path for a specific model's subdirectory.
func (c Config) ModelDir(modelName string) string {
	return fmt.Sprintf("%s/%s", c.ModelsDir, modelName)
}
