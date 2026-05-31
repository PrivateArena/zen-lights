package summarize

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLLMSummarizer_NoConfig(t *testing.T) {
	m := NewManager(Config{
		Algorithm: AlgoLLM,
	})

	_, err := m.Summarize("This is a test sentence.", 1, "en")
	if err == nil {
		t.Fatal("expected error when ModelPath is not configured, got nil")
	}
	if !strings.Contains(err.Error(), "ModelPath not configured") {
		t.Errorf("expected model path configuration error, got: %v", err)
	}
}

func TestLLMSummarizer_InvalidModelPaths(t *testing.T) {
	m := NewManager(Config{
		Algorithm: AlgoLLM,
		LLM: LLMConfig{
			ModelPath: "/nonexistent/model.onnx",
			ModelDir:  "/nonexistent/dir",
		},
	})

	_, err := m.Summarize("This is a test sentence.", 1, "en")
	if err == nil {
		t.Fatal("expected error with nonexistent paths, got nil")
	}
	if !strings.Contains(err.Error(), "load tokenizer error") && !strings.Contains(err.Error(), "no such file or directory") {
		t.Errorf("expected tokenizer load error, got: %v", err)
	}
}

func TestTokenizer_CacheAndGetOrLoad(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "tokenizer-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create dummy tokenizer.json
	tokenizerJSON := `{
		"model": {
			"vocab": {
				"<bos>": 2,
				"<eos>": 1,
				"<unk>": 3,
				"hello": 4
			}
		},
		"added_tokens": [
			{"content": "<start_of_turn>", "id": 105},
			{"content": "<end_of_turn>", "id": 106}
		]
	}`

	tokenPath := filepath.Join(tmpDir, "tokenizer.json")
	if err := os.WriteFile(tokenPath, []byte(tokenizerJSON), 0644); err != nil {
		t.Fatalf("failed to write dummy tokenizer.json: %v", err)
	}

	// First load (should read from disk)
	tok1, err := GetOrLoadTokenizer(tmpDir)
	if err != nil {
		t.Fatalf("failed to load tokenizer first time: %v", err)
	}
	if tok1 == nil {
		t.Fatal("expected non-nil tokenizer")
	}

	// Verify vocabulary loading
	if tok1.vocab["hello"] != 4 {
		t.Errorf("vocab parsing failed: expected hello -> 4, got %v", tok1.vocab["hello"])
	}

	// Second load (should hit the cache)
	tok2, err := GetOrLoadTokenizer(tmpDir)
	if err != nil {
		t.Fatalf("failed to load tokenizer second time: %v", err)
	}

	// Must be exactly the same pointer due to caching
	if tok1 != tok2 {
		t.Error("expected cached tokenizer instance to be identical to first load")
	}
}

func TestLLMSummarizer_RealModelIntegration(t *testing.T) {
	configPath := "../../config.json"

	m := NewManager(Config{})
	err := m.LoadConfig(configPath)
	if err != nil {
		t.Fatalf("failed to load config: %v", err)
	}

	m.mu.RLock()
	modelPath := m.cfg.LLM.ModelPath
	m.mu.RUnlock()

	// Check if real model files exist, if not skip
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Skipping real model integration test: model file not found at %s", modelPath)
	}

	text := "Gemma-3 is a state-of-the-art open large language model developed by Google. It is designed to be highly efficient, especially for lightweight inference tasks on local consumer-grade hardware. ONNX runtime enables high-performance inference across various platforms."
	tok, err := GetOrLoadTokenizer(m.cfg.LLM.ModelDir)
	if err != nil {
		t.Fatalf("load tokenizer error: %v", err)
	}
	if err := InitONNX(m.cfg.LLM.SharedLibPath, m.cfg.LLM.ModelPath, tok, m.cfg.LLM.NumLayers, m.cfg.LLM.VocabSize); err != nil {
		t.Fatalf("init onnx error: %v", err)
	}

	summaryText, err := SummarizeWithLLM(text, 80)
	if err != nil {
		t.Fatalf("SummarizeWithLLM failed: %v", err)
	}

	t.Logf("Generated Summary: %q", summaryText)
	if len(strings.TrimSpace(summaryText)) == 0 {
		t.Fatal("expected non-empty summary text")
	}
}

