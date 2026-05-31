package summarize

// SumBasicSummarizer implements the SumBasic frequency-based summarization algorithm.
type SumBasicSummarizer struct {
	Stemmer   Stemmer
	StopWords map[string]bool
}

// NewSumBasicSummarizer returns a new SumBasicSummarizer with defaults.
func NewSumBasicSummarizer(stemmer Stemmer, stopWords map[string]bool) *SumBasicSummarizer {
	return &SumBasicSummarizer{
		Stemmer:   stemmer,
		StopWords: stopWords,
	}
}

// Summarize extracts the top N sentences from a document using the SumBasic algorithm.
func (s *SumBasicSummarizer) Summarize(doc *Document, sentencesCount int) []Sentence {
	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return nil
	}

	ratings := s.computeRatings(sentences)
	return GetBestSentences(sentences, sentencesCount, ratings)
}

func (s *SumBasicSummarizer) getContentWords(sentence Sentence) []string {
	var contentWords []string
	for _, w := range sentence.Words {
		w = NormalizeWord(w)
		if s.StopWords[w] {
			continue
		}
		if s.Stemmer != nil {
			w = s.Stemmer.Stem(w)
		}
		if w != "" {
			contentWords = append(contentWords, w)
		}
	}
	return contentWords
}

func (s *SumBasicSummarizer) computeRatings(sentences []Sentence) map[int]float64 {
	var allContentWords []string
	sentencesWords := make([][]string, len(sentences))
	for idx, sent := range sentences {
		words := s.getContentWords(sent)
		sentencesWords[idx] = words
		allContentWords = append(allContentWords, words...)
	}

	totalWords := len(allContentWords)
	wordFreq := make(map[string]float64)
	if totalWords > 0 {
		counts := make(map[string]int)
		for _, w := range allContentWords {
			counts[w]++
		}
		for w, count := range counts {
			wordFreq[w] = float64(count) / float64(totalWords)
		}
	}

	ratings := make(map[int]float64)

	type activeSent struct {
		originalIdx int
		words       []string
	}
	var activeList []activeSent
	for idx, w := range sentencesWords {
		activeList = append(activeList, activeSent{
			originalIdx: idx,
			words:       w,
		})
	}

	selectionOrder := 0
	for len(activeList) > 0 {
		bestActiveIdx := 0
		maxAvgProb := -1.0

		for activeIdx, item := range activeList {
			avgProb := s.computeAverageProbability(wordFreq, item.words)
			if avgProb > maxAvgProb {
				maxAvgProb = avgProb
				bestActiveIdx = activeIdx
			}
		}

		bestSent := activeList[bestActiveIdx]
		activeList = append(activeList[:bestActiveIdx], activeList[bestActiveIdx+1:]...)

		ratings[bestSent.originalIdx] = float64(-selectionOrder)
		selectionOrder++

		s.updateTF(wordFreq, bestSent.words)
	}

	return ratings
}

func (s *SumBasicSummarizer) computeAverageProbability(wordFreq map[string]float64, words []string) float64 {
	if len(words) == 0 {
		return 0.0
	}
	var sum float64
	for _, w := range words {
		sum += wordFreq[w]
	}
	return sum / float64(len(words))
}

func (s *SumBasicSummarizer) updateTF(wordFreq map[string]float64, words []string) {
	for _, w := range words {
		wordFreq[w] = wordFreq[w] * wordFreq[w]
	}
}
