package translate

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/masacento/go-sentencepiece"
)

// Tokenizer wraps source/target SentencePiece models and vocab mapping for OPUS-MT.
type Tokenizer struct {
	srcProcessor *sentencepiece.Processor
	tgtProcessor *sentencepiece.Processor
	vocab        map[string]int64
	revVocab     map[int64]string
	padTokenID   int64
	eosTokenID   int64
}

// NewTokenizer creates a new Tokenizer.
func NewTokenizer(srcSPM, tgtSPM, vocabPath string, padTokenID, eosTokenID int64) (*Tokenizer, error) {
	srcProc, err := sentencepiece.NewProcessorFromPath(srcSPM)
	if err != nil {
		return nil, fmt.Errorf("load source SPM: %w", err)
	}

	var tgtProc *sentencepiece.Processor
	if tgtSPM != "" {
		tgtProc, err = sentencepiece.NewProcessorFromPath(tgtSPM)
		if err != nil {
			return nil, fmt.Errorf("load target SPM: %w", err)
		}
	}

	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("read vocab file: %w", err)
	}

	var rawVocab map[string]int64
	if err := json.Unmarshal(vocabData, &rawVocab); err != nil {
		return nil, fmt.Errorf("parse vocab JSON: %w", err)
	}

	// Create reverse vocab map for detokenization
	revVocab := make(map[int64]string, len(rawVocab))
	for k, v := range rawVocab {
		revVocab[v] = k
	}

	return &Tokenizer{
		srcProcessor: srcProc,
		tgtProcessor: tgtProc,
		vocab:        rawVocab,
		revVocab:     revVocab,
		padTokenID:   padTokenID,
		eosTokenID:   eosTokenID,
	}, nil
}

// Encode converts source text into a slice of vocabulary token IDs.
func (t *Tokenizer) Encode(text string) []int64 {
	tokens := t.srcProcessor.Encode(text)
	ids := make([]int64, 0, len(tokens)+1)

	for _, tok := range tokens {
		id, ok := t.vocab[tok.Text]
		if !ok {
			if unkID, exists := t.vocab["<unk>"]; exists {
				id = unkID
			} else {
				id = 1
			}
		}
		ids = append(ids, id)
	}

	ids = append(ids, t.eosTokenID)
	return ids
}

// Decode converts target vocabulary token IDs back into readable text.
func (t *Tokenizer) Decode(ids []int64) string {
	if t.tgtProcessor != nil {
		var spTokens []sentencepiece.Token
		for _, id := range ids {
			if id == t.padTokenID || id == t.eosTokenID {
				continue
			}
			piece, ok := t.revVocab[id]
			if !ok {
				continue
			}
			spTokens = append(spTokens, sentencepiece.Token{
				Text: piece,
			})
		}
		return t.tgtProcessor.DecodeTokens(spTokens)
	}

	var pieces []string
	for _, id := range ids {
		if id == t.padTokenID || id == t.eosTokenID {
			continue
		}
		piece, ok := t.revVocab[id]
		if ok {
			pieces = append(pieces, piece)
		}
	}

	joined := ""
	for i, p := range pieces {
		if strings.HasPrefix(p, " ") {
			if i > 0 {
				joined += " "
			}
			joined += strings.TrimPrefix(p, " ")
		} else {
			joined += p
		}
	}
	return joined
}
