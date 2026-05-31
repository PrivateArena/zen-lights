package stopwords

import (
	_ "embed"
	"strings"
)

//go:embed en.txt
var enRaw string

//go:embed zh.txt
var zhRaw string

//go:embed ja.txt
var jaRaw string

//go:embed vi.txt
var viRaw string

var registry = map[string]map[string]bool{}

func init() {
	registry["en"] = parse(enRaw)
	registry["zh"] = parse(zhRaw)
	registry["ja"] = parse(jaRaw)
	registry["vi"] = parse(viRaw)
}

func parse(raw string) map[string]bool {
	m := make(map[string]bool)
	lines := strings.Split(raw, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			m[strings.ToLower(line)] = true
		}
	}
	return m
}

// Get returns the stop words map for the given language.
// If the language is not supported, it returns an empty map.
func Get(lang string) map[string]bool {
	lang = strings.ToLower(strings.TrimSpace(lang))
	if m, ok := registry[lang]; ok {
		return m
	}
	// Fallback mappings
	switch lang {
	case "english":
		return registry["en"]
	case "chinese", "chinese_simplified", "chinese_traditional":
		return registry["zh"]
	case "japanese":
		return registry["ja"]
	case "vietnamese":
		return registry["vi"]
	}
	return make(map[string]bool)
}
