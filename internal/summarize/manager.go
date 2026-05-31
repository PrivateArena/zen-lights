package summarize

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

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
	// AlgoLLM uses local LLM (ONNX Gemma-3) for abstractive summarization.
	AlgoLLM Algorithm = "llm"
)

// LLMConfig holds configuration parameters for the local LLM ONNX session.
type LLMConfig struct {
	SharedLibPath        string
	ModelPath            string
	ModelDir             string
	NumLayers            int
	VocabSize            int64
	MaxTokensPerSentence int
}

// Config holds options for the summarization manager.
type Config struct {
	Algorithm Algorithm
	Stemmer   Stemmer
	LLM       LLMConfig
}

// Manager orchestrates parsing, word filtering, and executing summarizations.
type Manager struct {
	mu  sync.RWMutex
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

// LoadConfig loads the summarization configuration from a JSON file.
func (m *Manager) LoadConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read config file: %w", err)
	}

	var wrapper struct {
		Summarize struct {
			Algorithm            Algorithm `json:"algorithm"`
			SharedLibPath        string    `json:"shared_lib_path"`
			ModelPath            string    `json:"model_path"`
			ModelDir             string    `json:"model_dir"`
			NumLayers            int       `json:"num_layers"`
			VocabSize            int64     `json:"vocab_size"`
			MaxTokensPerSentence int       `json:"max_tokens_per_sentence"`
		} `json:"summarize"`
	}

	if err := json.Unmarshal(data, &wrapper); err != nil {
		return fmt.Errorf("unmarshal summarize config: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.cfg.Algorithm = wrapper.Summarize.Algorithm
	if m.cfg.Algorithm == "" {
		m.cfg.Algorithm = AlgoTextRank
	}

	m.cfg.LLM = LLMConfig{
		SharedLibPath:        wrapper.Summarize.SharedLibPath,
		ModelPath:            wrapper.Summarize.ModelPath,
		ModelDir:             wrapper.Summarize.ModelDir,
		NumLayers:            wrapper.Summarize.NumLayers,
		VocabSize:            wrapper.Summarize.VocabSize,
		MaxTokensPerSentence: wrapper.Summarize.MaxTokensPerSentence,
	}

	return nil
}

// Summarize splits text, filters stop words by language, and extracts the top N sentences.
func (m *Manager) Summarize(text string, count int, lang string) ([]string, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil
	}
	if count <= 0 {
		return nil, fmt.Errorf("sentence count must be positive")
	}

	m.mu.RLock()
	algo := m.cfg.Algorithm
	stemmer := m.cfg.Stemmer
	llmCfg := m.cfg.LLM
	m.mu.RUnlock()

	doc := ParseDocument(text)
	stopWords := stopwords.Get(lang)

	var summarizer Summarizer
	switch algo {
	case AlgoLuhn:
		summarizer = NewLuhnSummarizer(stemmer, stopWords)
	case AlgoTextRank:
		summarizer = NewTextRankSummarizer(stemmer, stopWords)
	case AlgoLexRank:
		summarizer = NewLexRankSummarizer(stemmer, stopWords)
	case AlgoLSA:
		summarizer = NewLsaSummarizer(stemmer, stopWords)
	case AlgoSumBasic:
		summarizer = NewSumBasicSummarizer(stemmer, stopWords)
	case AlgoLLM:
		if llmCfg.ModelPath == "" {
			return nil, fmt.Errorf("llm engine: ModelPath not configured")
		}
		tok, err := GetOrLoadTokenizer(llmCfg.ModelDir)
		if err != nil {
			return nil, fmt.Errorf("llm engine: load tokenizer error: %w", err)
		}
		if err := InitONNX(llmCfg.SharedLibPath, llmCfg.ModelPath, tok, llmCfg.NumLayers, llmCfg.VocabSize); err != nil {
			return nil, fmt.Errorf("llm engine: init onnx error: %w", err)
		}
		llmSummarizer := NewLLMSummarizer()
		if llmCfg.MaxTokensPerSentence > 0 {
			llmSummarizer.MaxTokensPerSentence = llmCfg.MaxTokensPerSentence
		}
		summarizer = llmSummarizer
	default:
		return nil, fmt.Errorf("unknown algorithm: %s", algo)
	}

	results := summarizer.Summarize(doc, count)

	var out []string
	for _, s := range results {
		out = append(out, s.Text)
	}
	return out, nil
}
