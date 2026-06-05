package svg

import (
	"embed"
	"encoding/json"
	"fmt"
	"strings"
)

//go:embed registry.json data
var EmbeddedAssets embed.FS

// Icon represents an SVG icon index entry.
type Icon struct {
	Name     string   `json:"name"`
	Dataset  string   `json:"dataset"`
	Filename string   `json:"filename"`
	Tags     []string `json:"tags"`
}

var (
	icons       []Icon
	initialized bool
)

// Init loads the embedded registry.json.
func Init() error {
	if initialized {
		return nil
	}
	data, err := EmbeddedAssets.ReadFile("registry.json")
	if err != nil {
		return fmt.Errorf("read embedded registry.json: %w", err)
	}
	if err := json.Unmarshal(data, &icons); err != nil {
		return fmt.Errorf("unmarshal registry: %w", err)
	}
	initialized = true
	return nil
}

// Search searches for an icon by query.
// It matches in the following priority:
// 1. Exact name match (case-insensitive)
// 2. Direct name match after stripping standard prefixes/suffixes
// 3. Name substring match
// 4. Exact tag match
// 5. Substring tag match
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

// ReadSVG reads the content of an icon's SVG file.
func ReadSVG(icon *Icon) (string, error) {
	if err := Init(); err != nil {
		return "", err
	}
	path := fmt.Sprintf("data/%s/%s", icon.Dataset, icon.Filename)
	data, err := EmbeddedAssets.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read embedded svg %s: %w", path, err)
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
	// Lucide icons use kebab-case
	return strings.Join(cleaned, "-")
}
