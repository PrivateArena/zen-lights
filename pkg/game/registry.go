package game

import (
	"fmt"
	"sort"
)

var registry = map[string]Detector{}

// Register adds a game Detector to the global registry.
// Call this inside your game package's init() function.
func Register(d Detector) {
	registry[d.Name()] = d
}

// Get retrieves a registered Detector by name.
func Get(name string) (Detector, error) {
	d, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown game %q — available: %v", name, Available())
	}
	return d, nil
}

// Available returns a sorted list of all registered game names.
func Available() []string {
	names := make([]string, 0, len(registry))
	for k := range registry {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}
