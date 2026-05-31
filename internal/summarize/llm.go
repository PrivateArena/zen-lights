package summarize

import (
	"strings"
)

// LLMSummarizer wraps the local ONNX inference engine to perform abstractive summarization.
type LLMSummarizer struct {
	MaxTokensPerSentence int
}

// NewLLMSummarizer creates a new LLMSummarizer instance.
func NewLLMSummarizer() *LLMSummarizer {
	return &LLMSummarizer{
		MaxTokensPerSentence: 80,
	}
}

// Summarize aggregates the input document, calls the local ONNX model, and returns a single summary sentence.
func (s *LLMSummarizer) Summarize(doc *Document, sentencesCount int) []Sentence {
	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return nil
	}

	var rawTexts []string
	for _, st := range sentences {
		rawTexts = append(rawTexts, st.Text)
	}
	text := strings.Join(rawTexts, " ")

	maxTokens := sentencesCount * s.MaxTokensPerSentence
	if maxTokens <= 0 {
		maxTokens = 80 // Sensible default
	}

	summaryText, err := SummarizeWithLLM(text, maxTokens)
	if err != nil {
		return nil
	}

	return []Sentence{
		{
			Text: strings.TrimSpace(summaryText),
		},
	}
}
