package summarize

// Sentence represents a single sentence in a document.
type Sentence struct {
	Text      string
	Words     []string
	IsHeading bool
}

// Paragraph represents a paragraph containing sentences and headings.
type Paragraph struct {
	Sentences []Sentence
	Headings  []Sentence
}

// Document represents a parsed document model containing paragraphs.
type Document struct {
	Paragraphs []Paragraph
}

// Sentences returns all sentences across all paragraphs in the document.
func (d *Document) Sentences() []Sentence {
	var sents []Sentence
	for _, p := range d.Paragraphs {
		sents = append(sents, p.Sentences...)
	}
	return sents
}

// Words returns all words across all sentences in the document.
func (d *Document) Words() []string {
	var words []string
	for _, p := range d.Paragraphs {
		for _, s := range p.Sentences {
			words = append(words, s.Words...)
		}
	}
	return words
}
