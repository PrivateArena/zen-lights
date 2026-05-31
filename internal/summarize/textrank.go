package summarize

import (
	"math"
)

// TextRankSummarizer implements the TextRank extractive summarization algorithm.
type TextRankSummarizer struct {
	Stemmer   Stemmer
	StopWords map[string]bool
	Epsilon   float64
	Damping   float64
}

// NewTextRankSummarizer returns a new TextRankSummarizer with defaults.
func NewTextRankSummarizer(stemmer Stemmer, stopWords map[string]bool) *TextRankSummarizer {
	return &TextRankSummarizer{
		Stemmer:   stemmer,
		StopWords: stopWords,
		Epsilon:   1e-4,
		Damping:   0.85,
	}
}

// Summarize extracts the top N sentences from a document using the TextRank algorithm.
func (t *TextRankSummarizer) Summarize(doc *Document, sentencesCount int) []Sentence {
	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return nil
	}

	sentencesAsWords := make([][]string, len(sentences))
	for i, s := range sentences {
		sentencesAsWords[i] = t.toWordsSet(s)
	}

	matrix := t.createMatrix(sentencesAsWords)
	ranks := PowerMethodTextRank(matrix, t.Epsilon)

	ratings := make(map[int]float64)
	for idx := range sentences {
		ratings[idx] = ranks[idx]
	}

	return GetBestSentences(sentences, sentencesCount, ratings)
}

func (t *TextRankSummarizer) toWordsSet(sentence Sentence) []string {
	var words []string
	for _, w := range sentence.Words {
		w = NormalizeWord(w)
		if t.StopWords[w] {
			continue
		}
		if t.Stemmer != nil {
			w = t.Stemmer.Stem(w)
		}
		if w != "" {
			words = append(words, w)
		}
	}
	return words
}

func rateSentencesEdge(words1, words2 []string) float64 {
	rank := 0
	for _, w1 := range words1 {
		count := 0
		for _, w2 := range words2 {
			if w1 == w2 {
				count++
			}
		}
		rank += count
	}

	if rank == 0 {
		return 0.0
	}

	if len(words1) == 0 || len(words2) == 0 {
		return 0.0
	}

	norm := math.Log(float64(len(words1))) + math.Log(float64(len(words2)))
	if math.Abs(norm) < 1e-9 {
		return float64(rank)
	}
	return float64(rank) / norm
}

func (t *TextRankSummarizer) createMatrix(sentencesAsWords [][]string) [][]float64 {
	n := len(sentencesAsWords)
	weights := make([][]float64, n)
	for i := range weights {
		weights[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			rating := rateSentencesEdge(sentencesAsWords[i], sentencesAsWords[j])
			weights[i][j] = rating
			weights[j][i] = rating
		}
	}

	// Normalize rows (weights sum per row)
	const zeroDivisionPrevention = 1e-7
	for i := 0; i < n; i++ {
		var rowSum float64
		for j := 0; j < n; j++ {
			rowSum += weights[i][j]
		}
		for j := 0; j < n; j++ {
			weights[i][j] /= (rowSum + zeroDivisionPrevention)
		}
	}

	// Apply damping
	matrix := make([][]float64, n)
	teleport := (1.0 - t.Damping) / float64(n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			matrix[i][j] = teleport + t.Damping*weights[i][j]
		}
	}

	return matrix
}
