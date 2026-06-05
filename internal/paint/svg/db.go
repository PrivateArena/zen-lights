package svg

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Icon represents an SVG icon index entry in the registry.
type Icon struct {
	Name     string   `json:"name"`
	Dataset  string   `json:"dataset"`
	Filename string   `json:"filename"`
	Tags     []string `json:"tags"`
}

var (
	icons        []Icon
	registryPath string
	baseDataPath string
	initOnce     sync.Once
	initErr      error
)

// Init loads the disk-based registry.json.
func Init() error {
	initOnce.Do(func() {
		// Detect absolute workspace path or fallback to relative path
		pathsToTry := []string{
			"/media/jang/home/Deve/zen-lights/assets/svg",
			"assets/svg",
		}

		var foundDir string
		for _, p := range pathsToTry {
			if info, err := os.Stat(p); err == nil && info.IsDir() {
				foundDir = p
				break
			}
		}

		if foundDir == "" {
			initErr = fmt.Errorf("assets/svg directory not found")
			return
		}

		registryPath = filepath.Join(foundDir, "registry.json")
		baseDataPath = filepath.Join(foundDir, "data")

		data, err := os.ReadFile(registryPath)
		if err != nil {
			initErr = fmt.Errorf("read registry.json: %w", err)
			return
		}

		if err := json.Unmarshal(data, &icons); err != nil {
			initErr = fmt.Errorf("unmarshal registry JSON: %w", err)
			return
		}
	})

	return initErr
}

// Search searches for an icon by query.
// Matches by exact name, stripped name, tag exact match, and tag substring.
func Search(query string) (*Icon, bool) {
	if err := Init(); err != nil {
		return nil, false
	}

	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil, false
	}

	// 1. Direct exact name match
	for _, icon := range icons {
		if strings.ToLower(icon.Name) == q {
			return &icon, true
		}
	}

	// 2. Direct name match after stripping standard prefixes/suffixes
	cleanedQ := cleanQuery(q)
	for _, icon := range icons {
		if strings.ToLower(icon.Name) == cleanedQ {
			return &icon, true
		}
	}

	// 3. Match if query contains icon name or vice versa
	for _, icon := range icons {
		iconName := strings.ToLower(icon.Name)
		if strings.Contains(cleanedQ, iconName) || strings.Contains(iconName, cleanedQ) {
			return &icon, true
		}
	}

	// 4. Exact match in tags
	for _, icon := range icons {
		for _, tag := range icon.Tags {
			if strings.ToLower(tag) == cleanedQ {
				return &icon, true
			}
		}
	}

	// 5. Substring match in tags
	for _, icon := range icons {
		for _, tag := range icon.Tags {
			if strings.Contains(strings.ToLower(tag), cleanedQ) {
				return &icon, true
			}
		}
	}

	return nil, false
}

// ReadSVG reads the content of an icon's SVG file from disk.
func ReadSVG(icon *Icon) (string, error) {
	if err := Init(); err != nil {
		return "", err
	}
	path := filepath.Join(baseDataPath, icon.Dataset, icon.Filename)
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read svg file %s: %w", path, err)
	}
	return string(data), nil
}

// cleanQuery removes common helper words to extract the core subject
func cleanQuery(q string) string {
	words := strings.Fields(q)
	var cleaned []string
	ignore := map[string]bool{
		"a":          true,
		"an":         true,
		"the":        true,
		"simple":     true,
		"clean":      true,
		"silhouette": true,
		"drawing":    true,
		"vector":     true,
		"icon":       true,
		"graphic":    true,
		"2d":         true,
		"shape":      true,
		"symbol":     true,
		"flat":       true,
	}
	for _, w := range words {
		if !ignore[w] {
			cleaned = append(cleaned, w)
		}
	}
	if len(cleaned) == 0 {
		return q
	}
	// Lucide icons use kebab-case filenames
	return strings.Join(cleaned, "-")
}
