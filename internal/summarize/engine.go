package summarize

import (
	"sort"
)

// Summarizer defines the standard interface for extractive summarizers.
type Summarizer interface {
	Summarize(doc *Document, sentencesCount int) []Sentence
}

type sentenceInfo struct {
	index    int
	sentence Sentence
	rating   float64
}

// GetBestSentences sorts the sentences by rating in descending order,
// extracts the top N, and returns them sorted by their original order in the document.
func GetBestSentences(sentences []Sentence, count int, ratings map[int]float64) []Sentence {
	var infos []sentenceInfo
	for idx, s := range sentences {
		infos = append(infos, sentenceInfo{
			index:    idx,
			sentence: s,
			rating:   ratings[idx],
		})
	}

	// Sort by rating descending. If ratings are equal, preserve original order.
	sort.Slice(infos, func(i, j int) bool {
		if infos[i].rating == infos[j].rating {
			return infos[i].index < infos[j].index
		}
		return infos[i].rating > infos[j].rating
	})

	limit := count
	if limit > len(infos) {
		limit = len(infos)
	}

	topInfos := infos[:limit]

	// Re-sort selected sentences by their original order in document
	sort.Slice(topInfos, func(i, j int) bool {
		return topInfos[i].index < topInfos[j].index
	})

	var result []Sentence
	for _, info := range topInfos {
		result = append(result, info.sentence)
	}
	return result
}
