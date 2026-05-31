package paint

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"

	"github.com/zen-lights/zen-lights/internal/paint/engine"
	"github.com/zen-lights/zen-lights/internal/paint/flux"
	"github.com/zen-lights/zen-lights/internal/paint/sdxl"
)

// Manager coordinates the lifecycle of active ONNX models and limits concurrency.
type Manager struct {
	Config       Config
	activeEngine engine.ImageEngine
	activeModel  string
	mu           sync.RWMutex
	sem          chan struct{}
}

// NewManager creates a new paint Manager.
func NewManager(cfg Config) *Manager {
	if cfg.MaxConcurrency <= 0 {
		cfg.MaxConcurrency = 1
	}
	return &Manager{
		Config: cfg,
		sem:    make(chan struct{}, cfg.MaxConcurrency),
	}
}

// LoadConfig reads the paint configuration block from the given JSON file.
func (m *Manager) LoadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			m.Config = DefaultConfig
			return nil
		}
		return fmt.Errorf("read config: %w", err)
	}

	var fullConfig struct {
		Paint *Config `json:"paint"`
	}
	if err := json.Unmarshal(data, &fullConfig); err != nil {
		return fmt.Errorf("unmarshal paint config: %w", err)
	}

	if fullConfig.Paint != nil {
		m.Config = *fullConfig.Paint
		if m.Config.MaxConcurrency <= 0 {
			m.Config.MaxConcurrency = 1
		}
		m.sem = make(chan struct{}, m.Config.MaxConcurrency)
	} else {
		m.Config = DefaultConfig
	}
	return nil
}

// AcquireSemaphore registers a concurrency slot. Returns false if server is busy.
func (m *Manager) AcquireSemaphore() bool {
	select {
	case m.sem <- struct{}{}:
		return true
	default:
		return false
	}
}

// ReleaseSemaphore releases a concurrency slot.
func (m *Manager) ReleaseSemaphore() {
	<-m.sem
}

// GetActive returns the active engine and loaded model name.
func (m *Manager) GetActive() (engine.ImageEngine, string) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeEngine, m.activeModel
}

// LoadEngine switches the active model to modelName.
func (m *Manager) LoadEngine(modelName string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeModel == modelName && m.activeEngine != nil {
		return nil
	}

	modelDir := m.Config.ModelDir(modelName)
	arch, err := readModelArch(modelDir)
	if err != nil {
		arch = "sdxl" // default fallback
	}

	opts := engine.Options{
		ExecutionProvider: m.Config.ExecutionProvider,
		NumThreads:        m.Config.NumThreads,
		OutputDir:         m.Config.OutputDir,
		OrtLib:            m.Config.OrtLibPath,
	}

	var eng engine.ImageEngine
	switch arch {
	case "sdxl", "sdxl-turbo", "lcm":
		eng = &sdxl.Engine{}
	case "flux", "bonsai":
		eng = &flux.Engine{}
	default:
		return fmt.Errorf("unknown model architecture %q", arch)
	}

	if err := eng.Initialize(modelDir, opts); err != nil {
		return fmt.Errorf("initialize engine: %w", err)
	}

	if m.activeEngine != nil {
		log.Printf("[paint] Closing active model %q before loading %q...", m.activeModel, modelName)
		_ = m.activeEngine.Close()
	}

	m.activeEngine = eng
	m.activeModel = modelName
	log.Printf("[paint] Loaded model: %s (%s)", modelName, arch)
	return nil
}

// Close releases the underlying ONNX engine resources.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.activeEngine != nil {
		log.Printf("[paint] Closing active model %q on shutdown...", m.activeModel)
		_ = m.activeEngine.Close()
		m.activeEngine = nil
		m.activeModel = ""
	}
}
