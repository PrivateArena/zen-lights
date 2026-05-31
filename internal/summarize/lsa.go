package summarize

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// LsaSummarizer implements the Latent Semantic Analysis extractive summarization algorithm.
type LsaSummarizer struct {
	Stemmer   Stemmer
	StopWords map[string]bool
}

const (
	lsaMinDimensions = 3
	lsaReductionRatio = 1.0
)

// NewLsaSummarizer returns a new LsaSummarizer with defaults.
func NewLsaSummarizer(stemmer Stemmer, stopWords map[string]bool) *LsaSummarizer {
	return &LsaSummarizer{
		Stemmer:   stemmer,
		StopWords: stopWords,
	}
}

// Summarize extracts the top N sentences from a document using the LSA algorithm.
func (l *LsaSummarizer) Summarize(doc *Document, sentencesCount int) []Sentence {
	sentences := doc.Sentences()
	if len(sentences) == 0 {
		return nil
	}

	dictionary := l.createDictionary(doc)
	if len(dictionary) == 0 {
		return nil
	}

	matrix := l.createMatrix(doc, dictionary)
	matrix = computeLsaTF(matrix, 0.4)

	// SVD decomposition using gonum
	r := len(matrix)
	c := len(matrix[0])
	data := make([]float64, r*c)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			data[i*c+j] = matrix[i][j]
		}
	}
	denseMat := mat.NewDense(r, c, data)

	var svd mat.SVD
	ok := svd.Factorize(denseMat, mat.SVDThin)
	if !ok {
		// SVD failed to converge, fallback to returning sentences as-is or first N
		limit := sentencesCount
		if limit > len(sentences) {
			limit = len(sentences)
		}
		return sentences[:limit]
	}

	// Retrieve singular values
	sigma := svd.Values(nil)

	// Retrieve V matrix (thin right singular vectors)
	var vMat mat.Dense
	svd.VTo(&vMat)

	ranks := l.computeRanks(sigma, &vMat)

	ratings := make(map[int]float64)
	for idx := range sentences {
		if idx < len(ranks) {
			ratings[idx] = ranks[idx]
		} else {
			ratings[idx] = 0.0
		}
	}

	return GetBestSentences(sentences, sentencesCount, ratings)
}

func (l *LsaSummarizer) createDictionary(doc *Document) map[string]int {
	words := doc.Words()
	uniqueWords := make(map[string]bool)

	for _, w := range words {
		w = NormalizeWord(w)
		if l.StopWords[w] {
			continue
		}
		if l.Stemmer != nil {
			w = l.Stemmer.Stem(w)
		}
		if w != "" {
			uniqueWords[w] = true
		}
	}

	dictionary := make(map[string]int)
	idx := 0
	for w := range uniqueWords {
		dictionary[w] = idx
		idx++
	}
	return dictionary
}

func (l *LsaSummarizer) createMatrix(doc *Document, dictionary map[string]int) [][]float64 {
	sentences := doc.Sentences()
	wordsCount := len(dictionary)
	sentencesCount := len(sentences)

	matrix := make([][]float64, wordsCount)
	for i := range matrix {
		matrix[i] = make([]float64, sentencesCount)
	}

	for col, s := range sentences {
		for _, w := range s.Words {
			stem := NormalizeWord(w)
			if l.Stemmer != nil {
				stem = l.Stemmer.Stem(stem)
			}
			if row, ok := dictionary[stem]; ok {
				matrix[row][col]++
			}
		}
	}

	return matrix
}

func computeLsaTF(matrix [][]float64, smooth float64) [][]float64 {
	r := len(matrix)
	if r == 0 {
		return matrix
	}
	c := len(matrix[0])

	maxWordFrequencies := make([]float64, c)
	for col := 0; col < c; col++ {
		var maxFreq float64
		for row := 0; row < r; row++ {
			if matrix[row][col] > maxFreq {
				maxFreq = matrix[row][col]
			}
		}
		maxWordFrequencies[col] = maxFreq
	}

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			maxFreq := maxWordFrequencies[col]
			if maxFreq != 0 {
				freq := matrix[row][col] / maxFreq
				matrix[row][col] = smooth + (1.0-smooth)*freq
			}
		}
	}

	return matrix
}

func (l *LsaSummarizer) computeRanks(sigma []float64, vMat *mat.Dense) []float64 {
	nRows, _ := vMat.Dims()

	dimensions := lsaMinDimensions
	dimVal := int(float64(len(sigma)) * lsaReductionRatio)
	if dimVal > dimensions {
		dimensions = dimVal
	}

	poweredSigma := make([]float64, len(sigma))
	for i, s := range sigma {
		if i < dimensions {
			poweredSigma[i] = s * s
		} else {
			poweredSigma[i] = 0.0
		}
	}

	ranks := make([]float64, nRows)
	for i := 0; i < nRows; i++ {
		var rank float64
		for j := 0; j < len(poweredSigma); j++ {
			vVal := vMat.At(i, j)
			rank += poweredSigma[j] * vVal * vVal
		}
		ranks[i] = math.Sqrt(rank)
	}

	return ranks
}
