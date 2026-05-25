package ocr

import (
	"fmt"
	"sync"
)

// LanguageProfile defines the assets for a specific language's OCR.
type LanguageProfile struct {
	ID           string
	RecModelPath string
	RecVocabPath string
	DetModelPath string // Optional, can share a common one
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
	opts.RecVocabPath = profile.RecVocabPath

	// Temporarily set env vars for New() to pick up
	// In a more refined version, New() should take these as parameters
	// but we'll stick to the current ocr.New structure for now.
	// We'll restore them if needed, but since this is a persistent server, 
	// we just set them.
	// Actually, let's modify ocr.New to be more flexible later.
	// For now, we manually set the environment or pass them if we refactor ocr.go
	
	// Refactoring internal/ocr/ocr.go slightly would be better to avoid Env var side effects.
	
	return nil, fmt.Errorf("initialization logic for %q needs ocr.New refactor", langID)
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
