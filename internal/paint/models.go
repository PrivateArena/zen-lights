package paint

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// modelMeta mirrors the model.json descriptor in each model subdirectory.
type modelMeta struct {
	Name         string            `json:"name"`
	Architecture string            `json:"architecture"`
	Files        map[string]string `json:"files"`
	DefaultSteps int               `json:"default_steps"`
	DefaultCFG   float32           `json:"default_cfg"`
	MaxWidth     int               `json:"max_width"`
	MaxHeight    int               `json:"max_height"`
	Description  string            `json:"description"`
}

// readModelArch reads only the architecture field from a model directory's model.json.
func readModelArch(modelDir string) (string, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "model.json"))
	if err != nil {
		return "", err
	}
	var m modelMeta
	if err := json.Unmarshal(data, &m); err != nil {
		return "", err
	}
	if m.Architecture == "" {
		return "", fmt.Errorf("model.json missing architecture field")
	}
	return m.Architecture, nil
}

// listModels returns names of all valid model subdirectories under modelsDir.
func listModels(modelsDir string) ([]string, error) {
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}
	var names []string
	for _, entry := range entries {
		if entry.IsDir() {
			// Check if model.json exists inside
			if _, err := os.Stat(filepath.Join(modelsDir, entry.Name(), "model.json")); err == nil {
				names = append(names, entry.Name())
			}
		}
	}
	return names, nil
}
