package ocr

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// LanguageProfile defines the assets for a specific language's OCR.
type LanguageProfile struct {
	ID           string `json:"id"`
	RecModelPath string `json:"rec_model_path"`
	RecVocabPath string `json:"rec_vocab_path"`
	DetModelPath string `json:"det_model_path"`
}

// Manager orchestrates multiple OCR clients for different languages.
type Manager struct {
	mu       sync.RWMutex
	clients  map[string]*Client
	profiles map[string]LanguageProfile
	options  Options
}

// NewManager creates a new multi-language OCR manager.
func NewManager(opts Options) *Manager {
	return &Manager{
		clients:  make(map[string]*Client),
		profiles: make(map[string]LanguageProfile),
		options:  opts,
	}
}

// LoadConfig loads language profiles from a JSON file.
func (m *Manager) LoadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read config file: %w", err)
	}

	var profiles []LanguageProfile
	if err := json.Unmarshal(data, &profiles); err != nil {
		return fmt.Errorf("unmarshal config: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, p := range profiles {
		m.profiles[p.ID] = p
	}
	return nil
}

// RegisterLanguage adds or updates a language profile.
func (m *Manager) RegisterLanguage(p LanguageProfile) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.profiles[p.ID] = p
}

// GetClient returns a cached OCR client for the given language,
// initializing it if necessary.
func (m *Manager) GetClient(langID string) (*Client, error) {
	m.mu.RLock()
	client, ok := m.clients[langID]
	profile, hasProfile := m.profiles[langID]
	m.mu.RUnlock()

	if ok {
		return client, nil
	}

	if !hasProfile {
		return nil, fmt.Errorf("language profile %q not registered", langID)
	}

	// Initialize new client for this language
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check again in case another goroutine initialized it
	if client, ok = m.clients[langID]; ok {
		return client, nil
	}

	opts := m.options
	opts.RecModelPath = profile.RecModelPath
	opts.RecVocabPath = profile.RecVocabPath
	if profile.DetModelPath != "" {
		opts.DetModelPath = profile.DetModelPath
	}

	client, err := New(opts)
	if err != nil {
		return nil, fmt.Errorf("init client for %q: %w", langID, err)
	}

	m.clients[langID] = client
	return client, nil
}

// Close releases all underlying OCR sessions.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, c := range m.clients {
		c.Close()
	}
	m.clients = make(map[string]*Client)
}
