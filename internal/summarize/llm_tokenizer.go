package summarize

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type SimpleTokenizer struct {
	vocab     map[string]int
	idToToken map[int]string
	index     map[byte][]string // Keys indexed by first byte
}

var (
	tokenizerCache   = make(map[string]*SimpleTokenizer)
	tokenizerCacheMu sync.Mutex
)

// GetOrLoadTokenizer retrieves a cached tokenizer for the given directory or loads it.
func GetOrLoadTokenizer(modelDir string) (*SimpleTokenizer, error) {
	tokenizerCacheMu.Lock()
	defer tokenizerCacheMu.Unlock()
	if tok, ok := tokenizerCache[modelDir]; ok {
		return tok, nil
	}
	tok, err := LoadTokenizer(modelDir)
	if err != nil {
		return nil, err
	}
	tokenizerCache[modelDir] = tok
	return tok, nil
}

func LoadTokenizer(modelDir string) (*SimpleTokenizer, error) {
	path := filepath.Join(modelDir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer: %w", err)
	}

	var tj struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			Content string `json:"content"`
			ID      int    `json:"id"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("parse tokenizer: %w", err)
	}

	t := &SimpleTokenizer{
		vocab:     tj.Model.Vocab,
		idToToken: make(map[int]string),
		index:     make(map[byte][]string),
	}

	for token, id := range t.vocab {
		t.idToToken[id] = token
	}
	for _, at := range tj.AddedTokens {
		t.vocab[at.Content] = at.ID
		t.idToToken[at.ID] = at.Content
	}

	// Prepare keys for greedy longest match, indexed by first byte
	for k := range t.vocab {
		if k != "" {
			b := k[0]
			t.index[b] = append(t.index[b], k)
		}
	}
	
	// Sort each bucket by length descending
	for b := range t.index {
		sort.Slice(t.index[b], func(i, j int) bool {
			return len(t.index[b][i]) > len(t.index[b][j])
		})
	}

	return t, nil
}

func (t *SimpleTokenizer) Encode(text string) ([]int64, error) {
	// Standard SentencePiece/Gemma-3 pre-processing
	text = strings.ReplaceAll(text, " ", "▁")
	
	var ids []int64
	// Always start with BOS for Gemma
	ids = append(ids, 2)

	i := 0
	for i < len(text) {
		matched := false
		b := text[i]
		if keys, ok := t.index[b]; ok {
			for _, key := range keys {
				if strings.HasPrefix(text[i:], key) {
					ids = append(ids, int64(t.vocab[key]))
					i += len(key)
					matched = true
					break
				}
			}
		}

		if !matched {
			// Byte fallback
			b := text[i]
			byteToken := fmt.Sprintf("<0x%02X>", b)
			if id, ok := t.vocab[byteToken]; ok {
				ids = append(ids, int64(id))
			} else {
				// Truly unknown, skip or use <unk> (3)
				ids = append(ids, 3)
			}
			i++
		}
	}

	return ids, nil
}

func (t *SimpleTokenizer) Decode(ids []int64) (string, error) {
	var result strings.Builder
	var byteBuffer []byte

	flushBytes := func() {
		if len(byteBuffer) > 0 {
			result.Write(byteBuffer)
			byteBuffer = nil
		}
	}

	for _, id := range ids {
		token, ok := t.idToToken[int(id)]
		if !ok {
			continue
		}

		// Handle byte fallback tokens: <0x..>
		if strings.HasPrefix(token, "<0x") && strings.HasSuffix(token, ">") && len(token) == 6 {
			var b byte
			_, err := fmt.Sscanf(token, "<0x%02X>", &b)
			if err == nil {
				byteBuffer = append(byteBuffer, b)
				continue
			}
		}

		flushBytes()

		// Skip special tokens in decoding
		if (strings.HasPrefix(token, "<") && strings.HasSuffix(token, ">")) || id < 5 {
			continue
		}

		// Restore space marker
		token = strings.ReplaceAll(token, "▁", " ")
		result.WriteString(token)
	}
	flushBytes()

	return result.String(), nil
}
