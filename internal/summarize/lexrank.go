package summarize

import (
	"math"
)

// LexRankSummarizer implements the LexRank extractive summarization algorithm.
type LexRankSummarizer struct {
	Stemmer   Stemmer
	StopWords map[string]bool
	Threshold float64
	Epsilon   float64
}

// NewLexRankSummarizer returns a new LexRankSummarizer with defaults.
func NewLexRankSummarizer(stemmer Stemmer, stopWords map[string]bool) *LexRankSummarizer {
	return &LexRankSummarizer{
		Stemmer:   stemmer,
		StopWords: stopWords,
		Threshold: 0.1,
		Epsilon:   0.1,
	}
}

// Summarize extracts the top N sentences from a document using the LexRank algorithm.
func (l *LexRankSummarizer) Summarize(doc *Document, sentencesCount int) []Sentence {
	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return nil
	}

	sentencesWords := make([][]string, len(sentences))
	for i, s := range sentences {
		sentencesWords[i] = l.toWordsSet(s)
	}

	tfMetrics := computeTF(sentencesWords)
	idfMetrics := computeIDF(sentencesWords)

	matrix := l.createMatrix(sentencesWords, tfMetrics, idfMetrics)
	scores := PowerMethodLexRank(matrix, l.Epsilon)

	ratings := make(map[int]float64)
	for idx := range sentences {
		ratings[idx] = scores[idx]
	}

	return GetBestSentences(sentences, sentencesCount, ratings)
}

func (l *LexRankSummarizer) toWordsSet(sentence Sentence) []string {
	var words []string
	for _, w := range sentence.Words {
		w = NormalizeWord(w)
		if l.StopWords[w] {
			continue
		}
		if l.Stemmer != nil {
			w = l.Stemmer.Stem(w)
		}
		if w != "" {
			words = append(words, w)
		}
	}
	return words
}

func computeTF(sentencesWords [][]string) []map[string]float64 {
	var tfMetrics []map[string]float64
	for _, words := range sentencesWords {
		counts := make(map[string]int)
		maxTF := 1
		for _, w := range words {
			counts[w]++
			if counts[w] > maxTF {
				maxTF = counts[w]
			}
		}

		metrics := make(map[string]float64)
		for term, count := range counts {
			metrics[term] = float64(count) / float64(maxTF)
		}
		tfMetrics = append(tfMetrics, metrics)
	}
	return tfMetrics
}

func computeIDF(sentencesWords [][]string) map[string]float64 {
	idfMetrics := make(map[string]float64)
	sentencesCount := len(sentencesWords)

	// Count sentences containing each term in a single pass
	docFreqs := make(map[string]int)
	for _, words := range sentencesWords {
		seen := make(map[string]bool)
		for _, w := range words {
			if !seen[w] {
				seen[w] = true
				docFreqs[w]++
			}
		}
	}

	for term, nJ := range docFreqs {
		idfMetrics[term] = math.Log(float64(sentencesCount) / float64(1+nJ))
	}

	return idfMetrics
}

func cosineSimilarity(words1, words2 []string, tf1, tf2 map[string]float64, idfMetrics map[string]float64) float64 {
	seen1 := make(map[string]bool)
	for _, w := range words1 {
		seen1[w] = true
	}

	seen2 := make(map[string]bool)
	for _, w := range words2 {
		seen2[w] = true
	}

	numerator := 0.0
	for w := range seen1 {
		if seen2[w] {
			numerator += tf1[w] * tf2[w] * math.Pow(idfMetrics[w], 2)
		}
	}

	denominator1 := 0.0
	for w := range seen1 {
		denominator1 += math.Pow(tf1[w]*idfMetrics[w], 2)
	}

	denominator2 := 0.0
	for w := range seen2 {
		denominator2 += math.Pow(tf2[w]*idfMetrics[w], 2)
	}

	if denominator1 > 0 && denominator2 > 0 {
		return numerator / (math.Sqrt(denominator1) * math.Sqrt(denominator2))
	}
	return 0.0
}

func (l *LexRankSummarizer) createMatrix(sentencesWords [][]string, tfMetrics []map[string]float64, idfMetrics map[string]float64) [][]float64 {
	n := len(sentencesWords)
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}
	degrees := make([]float64, n)

	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			sim := cosineSimilarity(sentencesWords[row], sentencesWords[col], tfMetrics[row], tfMetrics[col], idfMetrics)
			if sim > l.Threshold {
				matrix[row][col] = 1.0
				degrees[row]++
			} else {
				matrix[row][col] = 0.0
			}
		}
	}

	for row := 0; row < n; row++ {
		deg := degrees[row]
		if deg == 0 {
			deg = 1
		}
		for col := 0; col < n; col++ {
			matrix[row][col] /= deg
		}
	}

	return matrix
}
