package translate

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
)

// Manager orchestrates translation requests and manages engine lifetimes.
type Manager struct {
	mu             sync.RWMutex
	config         Config
	profiles       map[string]TranslationProfile
	offlineEngines map[string]*OfflineEngine
	onlineEngine   *OnlineEngine
}

// NewManager creates a new translation Manager.
func NewManager(cfg Config) *Manager {
	profiles := make(map[string]TranslationProfile)
	for _, p := range cfg.Profiles {
		profiles[p.ID] = p
	}

	return &Manager{
		config:         cfg,
		profiles:       profiles,
		offlineEngines: make(map[string]*OfflineEngine),
	}
}

// LoadConfig loads translation configurations from a JSON file.
func (m *Manager) LoadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read config file: %w", err)
	}

	var wrapper struct {
		Translation Config `json:"translation"`
	}
	if err := json.Unmarshal(data, &wrapper); err != nil {
		return fmt.Errorf("unmarshal translation config: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.config = wrapper.Translation
	if m.config.Mode == "" {
		m.config.Mode = ModeAuto
	}
	if m.config.MaxTokens <= 0 {
		m.config.MaxTokens = 128
	}

	m.profiles = make(map[string]TranslationProfile)
	for _, p := range m.config.Profiles {
		m.profiles[p.ID] = p
	}

	return nil
}

// RegisterProfile registers or updates a translation profile.
func (m *Manager) RegisterProfile(p TranslationProfile) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.profiles[p.ID] = p
}

// HasProfile checks if a translation profile is registered.
func (m *Manager) HasProfile(profileID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, ok := m.profiles[profileID]
	return ok
}

// GetOfflineEngine retrieves or initializes the offline translation engine for the given profile.
func (m *Manager) GetOfflineEngine(profileID string) (*OfflineEngine, error) {
	m.mu.RLock()
	eng, ok := m.offlineEngines[profileID]
	prof, hasProfile := m.profiles[profileID]
	m.mu.RUnlock()

	if ok {
		return eng, nil
	}

	if !hasProfile {
		return nil, fmt.Errorf("translation profile %q not found: %w", profileID, ErrUnsupportedLanguagePair)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if eng, ok = m.offlineEngines[profileID]; ok {
		return eng, nil
	}

	log.Printf("⚙️ Translation: Loading offline model for profile %q...", profileID)
	eng, err := NewOfflineEngine(prof, m.config.MaxTokens)
	if err != nil {
		return nil, fmt.Errorf("init offline engine for %q: %w", profileID, err)
	}
	log.Printf("❇️ Translation: Successfully loaded offline model for profile %q", profileID)

	m.offlineEngines[profileID] = eng
	return eng, nil
}

// GetOnlineEngine retrieves or initializes the online translation engine.
func (m *Manager) GetOnlineEngine() *OnlineEngine {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.onlineEngine == nil {
		m.onlineEngine = NewOnlineEngine()
	}
	return m.onlineEngine
}

// Translate performs translation using the configured engine/mode routing.
func (m *Manager) Translate(ctx context.Context, text, srcLang, tgtLang string) (string, error) {
	if text == "" {
		return "", nil
	}

	profileID := fmt.Sprintf("%s-%s", srcLang, tgtLang)

	mode := m.config.Mode
	if mode == "" {
		mode = ModeAuto
	}

	switch mode {
	case ModeOnline:
		return m.GetOnlineEngine().Translate(ctx, text, srcLang, tgtLang)

	case ModeOffline:
		offEng, err := m.GetOfflineEngine(profileID)
		if err != nil {
			return "", err
		}
		return offEng.Translate(ctx, text, srcLang, tgtLang)

	case ModeAuto:
		if m.HasProfile(profileID) {
			offEng, err := m.GetOfflineEngine(profileID)
			if err == nil {
				res, err := offEng.Translate(ctx, text, srcLang, tgtLang)
				if err == nil {
					return res, nil
				}
				log.Printf("Warning: offline translation failed for %q, falling back to online: %v", text, err)
			} else {
				log.Printf("Warning: failed to initialize offline engine for %q, falling back to online: %v", profileID, err)
			}
		}

		return m.GetOnlineEngine().Translate(ctx, text, srcLang, tgtLang)

	default:
		return "", fmt.Errorf("unknown translation mode: %s", mode)
	}
}

// Close releases all loaded translation sessions. Always defer this.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, eng := range m.offlineEngines {
		if err := eng.Close(); err != nil {
			log.Printf("Error closing offline engine %q: %v", id, err)
		}
	}
	m.offlineEngines = make(map[string]*OfflineEngine)

	if m.onlineEngine != nil {
		m.onlineEngine.Close()
		m.onlineEngine = nil
	}
}
