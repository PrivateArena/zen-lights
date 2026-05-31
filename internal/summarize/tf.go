package summarize

import (
	"math"
	"sort"
)

// TFModel represents a term frequency model for a document or text block.
type TFModel struct {
	Terms        map[string]int
	MaxFrequency int
}

// NewTFModel constructs a new TFModel from a list of terms/words.
func NewTFModel(words []string, stemmer Stemmer) *TFModel {
	m := &TFModel{
		Terms:        make(map[string]int),
		MaxFrequency: 1,
	}

	for _, w := range words {
		w = NormalizeWord(w)
		if stemmer != nil {
			w = stemmer.Stem(w)
		}
		if w != "" {
			m.Terms[w]++
		}
	}

	maxFreq := 1
	for _, freq := range m.Terms {
		if freq > maxFreq {
			maxFreq = freq
		}
	}
	m.MaxFrequency = maxFreq

	return m
}

// Magnitude returns the L2 norm (magnitude) of the term vector.
func (m *TFModel) Magnitude() float64 {
	var sumSquares float64
	for _, freq := range m.Terms {
		sumSquares += float64(freq * freq)
	}
	return math.Sqrt(sumSquares)
}

// TermsList returns all terms in this model.
func (m *TFModel) TermsList() []string {
	terms := make([]string, 0, len(m.Terms))
	for t := range m.Terms {
		terms = append(terms, t)
	}
	return terms
}

// TermFrequency returns the raw frequency of a term.
func (m *TFModel) TermFrequency(term string) int {
	return m.Terms[term]
}

// NormalizedTermFrequency returns the normalized term frequency.
func (m *TFModel) NormalizedTermFrequency(term string, smooth float64) float64 {
	freq := float64(m.TermFrequency(term))
	if m.MaxFrequency == 0 {
		return smooth
	}
	return smooth + (1.0-smooth)*(freq/float64(m.MaxFrequency))
}

type termFreq struct {
	term string
	freq int
}

// MostFrequentTerms returns the top N terms sorted by frequency descending.
// If n <= 0, returns all terms.
func (m *TFModel) MostFrequentTerms(n int) []string {
	var list []termFreq
	for t, f := range m.Terms {
		list = append(list, termFreq{t, f})
	}

	// Sort descending by frequency. Tie-break alphabetically for determinism.
	sort.Slice(list, func(i, j int) bool {
		if list[i].freq == list[j].freq {
			return list[i].term < list[j].term
		}
		return list[i].freq > list[j].freq
	})

	var result []string
	limit := len(list)
	if n > 0 && n < limit {
		limit = n
	}
	for i := 0; i < limit; i++ {
		result = append(result, list[i].term)
	}
	return result
}
