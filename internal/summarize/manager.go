package summarize

import (
	"fmt"
	"strings"

	"github.com/zen-lights/zen-lights/internal/summarize/stopwords"
)

// Algorithm defines the available summarization engines.
type Algorithm string

const (
	// AlgoLuhn uses the Luhn significance-percentage and chunk-rating algorithm.
	AlgoLuhn Algorithm = "luhn"
	// AlgoTextRank uses PageRank on a word-overlap sentence similarity graph.
	AlgoTextRank Algorithm = "textrank"
	// AlgoLexRank uses PageRank on a TF-IDF cosine similarity sentence graph.
	AlgoLexRank Algorithm = "lexrank"
	// AlgoLSA uses Singular Value Decomposition (SVD) representation for semantic ranking.
	AlgoLSA Algorithm = "lsa"
	// AlgoSumBasic uses word probability-dampening greedily.
	AlgoSumBasic Algorithm = "sumbasic"
)

// Config holds options for the summarization manager.
type Config struct {
	Algorithm Algorithm
	Stemmer   Stemmer
}

// Manager orchestrates parsing, word filtering, and executing summarizations.
type Manager struct {
	cfg Config
}

// NewManager constructs a new Manager instance.
func NewManager(cfg Config) *Manager {
	if cfg.Algorithm == "" {
		cfg.Algorithm = AlgoTextRank // Default to TextRank
	}
	if cfg.Stemmer == nil {
		cfg.Stemmer = NullStemmer{}
	}
	return &Manager{cfg: cfg}
}

// Summarize splits text, filters stop words by language, and extracts the top N sentences.
func (m *Manager) Summarize(text string, count int, lang string) ([]string, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	if count <= 0 {
		return nil, fmt.Errorf("sentence count must be positive")
	}

	doc := ParseDocument(text)
	stopWords := stopwords.Get(lang)

	var summarizer Summarizer
	switch m.cfg.Algorithm {
	case AlgoLuhn:
		summarizer = NewLuhnSummarizer(m.cfg.Stemmer, stopWords)
	case AlgoTextRank:
		summarizer = NewTextRankSummarizer(m.cfg.Stemmer, stopWords)
	case AlgoLexRank:
		summarizer = NewLexRankSummarizer(m.cfg.Stemmer, stopWords)
	case AlgoLSA:
		summarizer = NewLsaSummarizer(m.cfg.Stemmer, stopWords)
	case AlgoSumBasic:
		summarizer = NewSumBasicSummarizer(m.cfg.Stemmer, stopWords)
	default:
		return nil, fmt.Errorf("unknown algorithm: %s", m.cfg.Algorithm)
	}

	results := summarizer.Summarize(doc, count)

	var out []string
	for _, s := range results {
		out = append(out, s.Text)
	}
	return out, nil
}
