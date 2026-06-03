package summarize

import (
	"strings"
	"unicode"
)

// TokenizeSentences splits text into individual sentences.
func TokenizeSentences(text string) []string {
	// Replace all newlines with spaces to handle wrapped text
	text = strings.ReplaceAll(text, "\r\n", " ")
	text = strings.ReplaceAll(text, "\n", " ")

	var sents []string
	var current strings.Builder
	runes := []rune(text)
	n := len(runes)

	for i := 0; i < n; i++ {
		r := runes[i]
		current.WriteRune(r)

		isDelim := false
		switch r {
		case '?', '!', '。', '？', '！':
			isDelim = true
		case '.':
			isDelim = true
			if i+1 < n {
				next := runes[i+1]
				// If followed by digit, not sentence end (decimal number)
				if unicode.IsDigit(next) {
					isDelim = false
				}
			}
			// Check for common abbreviations before this dot
			currStr := current.String()
			for _, abbr := range []string{"Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "e.g.", "i.e.", "a.m.", "p.m."} {
				if strings.HasSuffix(strings.ToLower(currStr), " "+strings.ToLower(abbr)) || strings.ToLower(currStr) == strings.ToLower(abbr) {
					isDelim = false
					break
				}
			}
		}

		if isDelim {
			// CJK delimiters do not need spaces after them to split sentences
			isCJKDelim := r == '。' || r == '？' || r == '！'
			if isCJKDelim || i+1 == n || unicode.IsSpace(runes[i+1]) {
				trimmed := strings.TrimSpace(current.String())
				if trimmed != "" {
					sents = append(sents, trimmed)
				}
				current.Reset()
			}
		}
	}

	trimmed := strings.TrimSpace(current.String())
	if trimmed != "" {
		sents = append(sents, trimmed)
	}

	return sents
}

// TokenizeWords splits a sentence text into words.
// It treats CJK characters individually and groups Latin/numeric/Vietnamese letters.
func TokenizeWords(sentence string) []string {
	var words []string
	var current strings.Builder
	runes := []rune(sentence)

	flush := func() {
		if current.Len() > 0 {
			words = append(words, current.String())
			current.Reset()
		}
	}

	for _, r := range runes {
		if isCJK(r) {
			flush()
			words = append(words, string(r))
		} else if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' || r == '-' {
			current.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()

	return words
}

func isCJK(r rune) bool {
	// Chinese characters (Unified Ideographs)
	if r >= 0x4e00 && r <= 0x9fff {
		return true
	}
	// Japanese Hiragana
	if r >= 0x3040 && r <= 0x309f {
		return true
	}
	// Japanese Katakana
	if r >= 0x30a0 && r <= 0x30ff {
		return true
	}
	// Korean Hangul
	if r >= 0xac00 && r <= 0xd7af {
		return true
	}
	return false
}

// ParseDocument parses a raw text string into a Document model.
func ParseDocument(text string) *Document {
	paragraphs := strings.Split(text, "\n\n")
	var doc Document

	for _, pText := range paragraphs {
		pText = strings.TrimSpace(pText)
		if pText == "" {
			continue
		}

		var para Paragraph
		sents := TokenizeSentences(pText)
		for _, sText := range sents {
			sent := Sentence{
				Text:      sText,
				Words:     TokenizeWords(sText),
				IsHeading: false, // Defaulting to false, extensible
			}
			para.Sentences = append(para.Sentences, sent)
		}

		if len(para.Sentences) > 0 {
			doc.Paragraphs = append(doc.Paragraphs, para)
		}
	}

	return &doc
}
