package summarize

import "strings"

// Stemmer defines the interface for stemming words.
type Stemmer interface {
	Stem(word string) string
}

// NullStemmer is a no-op stemmer that returns the word as-is.
type NullStemmer struct{}

// Stem returns the word as-is.
func (n NullStemmer) Stem(word string) string {
	return word
}

// NormalizeWord lowercases and trims whitespace/punctuation from a word.
func NormalizeWord(word string) string {
	return strings.ToLower(strings.TrimSpace(word))
}
