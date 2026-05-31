package summarize

// LuhnSummarizer implements the Luhn extractive summarization algorithm.
type LuhnSummarizer struct {
	Stemmer               Stemmer
	StopWords             map[string]bool
	MaxGapSize            int
	SignificantPercentage float64
}

// NewLuhnSummarizer returns a new LuhnSummarizer initialized with defaults.
func NewLuhnSummarizer(stemmer Stemmer, stopWords map[string]bool) *LuhnSummarizer {
	return &LuhnSummarizer{
		Stemmer:               stemmer,
		StopWords:             stopWords,
		MaxGapSize:            4,
		SignificantPercentage: 1.0,
	}
}

// Summarize extracts the top N sentences from a document using the Luhn algorithm.
func (l *LuhnSummarizer) Summarize(doc *Document, sentencesCount int) []Sentence {
	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return nil
	}

	significantStems := l.getSignificantWords(doc)
	ratings := make(map[int]float64)

	for idx, s := range sentences {
		ratings[idx] = l.rateSentence(s, significantStems)
	}

	return GetBestSentences(sentences, sentencesCount, ratings)
}

func (l *LuhnSummarizer) getSignificantWords(doc *Document) map[string]bool {
	words := doc.Words()
	var filtered []string
	for _, w := range words {
		w = NormalizeWord(w)
		if l.StopWords[w] {
			continue
		}
		if l.Stemmer != nil {
			w = l.Stemmer.Stem(w)
		}
		if w != "" {
			filtered = append(filtered, w)
		}
	}

	model := NewTFModel(filtered, nil) // words already stemmed
	bestCount := int(float64(len(filtered)) * l.SignificantPercentage)
	freqTerms := model.MostFrequentTerms(bestCount)

	significantStems := make(map[string]bool)
	for _, t := range freqTerms {
		if model.TermFrequency(t) > 1 {
			significantStems[t] = true
		}
	}
	return significantStems
}

func (l *LuhnSummarizer) rateSentence(sentence Sentence, significantStems map[string]bool) float64 {
	ratings := l.getChunkRatings(sentence, significantStems)
	if len(ratings) == 0 {
		return 0
	}
	maxRating := 0.0
	for _, r := range ratings {
		if r > maxRating {
			maxRating = r
		}
	}
	return maxRating
}

func (l *LuhnSummarizer) getChunkRatings(sentence Sentence, significantStems map[string]bool) []float64 {
	var chunks [][]int
	inChunk := false

	for _, w := range sentence.Words {
		stem := NormalizeWord(w)
		if l.Stemmer != nil {
			stem = l.Stemmer.Stem(stem)
		}

		isSig := significantStems[stem]

		if isSig && !inChunk {
			inChunk = true
			chunks = append(chunks, []int{1})
		} else if inChunk {
			val := 0
			if isSig {
				val = 1
			}
			chunks[len(chunks)-1] = append(chunks[len(chunks)-1], val)
		}

		// check for end of chunk (last MaxGapSize elements are 0)
		if len(chunks) > 0 {
			lastChunk := chunks[len(chunks)-1]
			if len(lastChunk) >= l.MaxGapSize {
				allZeros := true
				for i := len(lastChunk) - l.MaxGapSize; i < len(lastChunk); i++ {
					if lastChunk[i] != 0 {
						allZeros = false
						break
					}
				}
				if allZeros {
					inChunk = false
				}
			}
		}
	}

	var ratings []float64
	for _, chunk := range chunks {
		ratings = append(ratings, l.getChunkRating(chunk))
	}
	return ratings
}

func (l *LuhnSummarizer) getChunkRating(chunk []int) float64 {
	chunk = removeTrailingZeros(chunk)
	wordsCount := len(chunk)
	if wordsCount == 0 {
		return 0
	}

	significantWords := 0
	for _, val := range chunk {
		significantWords += val
	}

	if significantWords == 1 {
		return 0
	}

	return float64(significantWords*significantWords) / float64(wordsCount)
}

func removeTrailingZeros(chunk []int) []int {
	idx := len(chunk) - 1
	for idx >= 0 && chunk[idx] == 0 {
		idx--
	}
	return chunk[:idx+1]
}
