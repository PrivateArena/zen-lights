package summarize

import (
	"strings"
	"testing"

	"github.com/zen-lights/zen-lights/internal/summarize/stopwords"
)

func TestTokenizer(t *testing.T) {
	// English sentence tokenizer with abbreviations
	text := "Mr. Green went to the store. He bought a newspaper for $1.50. Then he went home."
	sents := TokenizeSentences(text)
	if len(sents) != 3 {
		t.Errorf("expected 3 sentences, got %d: %v", len(sents), sents)
	}

	// CJK sentence tokenizer (without space)
	cjkText := "今天天气很好。我们出去玩吧！你觉得怎么样？"
	cjkSents := TokenizeSentences(cjkText)
	if len(cjkSents) != 3 {
		t.Errorf("expected 3 CJK sentences, got %d: %v", len(cjkSents), cjkSents)
	}

	// Word tokenizer for English
	wordsEn := TokenizeWords("Mr. Green went home.")
	if len(wordsEn) < 3 {
		t.Errorf("expected word tokens, got %v", wordsEn)
	}

	// Word tokenizer for Chinese/Japanese (character level)
	wordsZh := TokenizeWords("今天")
	if len(wordsZh) != 2 || wordsZh[0] != "今" || wordsZh[1] != "天" {
		t.Errorf("expected character level splitting for Chinese, got %v", wordsZh)
	}
}

func TestStopwords(t *testing.T) {
	// Test English stop words loading
	stopEn := stopwords.Get("en")
	if !stopEn["the"] {
		t.Error("expected 'the' to be an English stop word")
	}

	// Test Vietnamese stop word
	stopVi := stopwords.Get("vi")
	if !stopVi["của"] {
		t.Error("expected 'của' to be a Vietnamese stop word")
	}
}

func TestSummarizers(t *testing.T) {
	sampleText := `Extractive text summarization is a popular technique to summarize long documents. 
It works by choosing the most important sentences from the original text. 
Different algorithms use different heuristics to score the sentences. 
Some algorithms, like Luhn, use word frequency. 
Other algorithms, like TextRank or LexRank, use graph-based centrality. 
SumBasic is frequency-based and uses probability-dampening. 
LSA is vector-space-based and uses singular value decomposition to capture concepts. 
We want to verify that all of these algorithms produce reasonable output in Go.`

	algos := []Algorithm{AlgoLuhn, AlgoTextRank, AlgoLexRank, AlgoLSA, AlgoSumBasic}

	for _, algo := range algos {
		t.Run(string(algo), func(t *testing.T) {
			m := NewManager(Config{Algorithm: algo})
			summary, err := m.Summarize(sampleText, 3, "en")
			if err != nil {
				t.Fatalf("failed to summarize with %s: %v", algo, err)
			}
			if len(summary) != 3 {
				t.Errorf("expected 3 sentences, got %d: %v", len(summary), summary)
			}
			for _, s := range summary {
				if strings.TrimSpace(s) == "" {
					t.Error("empty sentence returned in summary")
				}
			}
		})
	}
}

func TestMultilingualSummaries(t *testing.T) {
	// Chinese sample text
	zhText := `文本摘要是自然语言处理中的一个重要任务。它旨在将长文本缩短，同时保留其核心信息。
不同的提取式算法使用不同的启发式方法对句子进行评分。
例如，Luhn算法利用词频来选择显著的句子。
而TextRank或LexRank算法则利用基于图的中心性来评估句子的重要性。
这对于高效率地获取文档的核心思想非常有用。`

	// Vietnamese sample text
	viText := `Tóm tắt văn bản là một nhiệm vụ quan trọng trong xử lý ngôn ngữ tự nhiên. Nó giúp rút ngắn văn bản dài nhưng vẫn giữ lại thông tin cốt lõi.
Các thuật toán trích xuất khác nhau sử dụng các phương pháp heuristics khác nhau để chấm điểm câu.
Ví dụ, thuật toán Luhn sử dụng tần suất từ để chọn các câu nổi bật.
Thuật toán TextRank hoặc LexRank sử dụng tính trung tâm dựa trên đồ thị để đánh giá tầm quan trọng của câu.
Điều này cực kỳ hữu ích để nắm bắt nhanh ý chính của tài liệu.`

	tests := []struct {
		name string
		text string
		lang string
	}{
		{"Chinese", zhText, "zh"},
		{"Vietnamese", viText, "vi"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			for _, algo := range []Algorithm{AlgoLuhn, AlgoTextRank, AlgoLexRank, AlgoLSA, AlgoSumBasic} {
				m := NewManager(Config{Algorithm: algo})
				summary, err := m.Summarize(tc.text, 2, tc.lang)
				if err != nil {
					t.Fatalf("failed to summarize in %s with %s: %v", tc.lang, algo, err)
				}
				if len(summary) != 2 {
					t.Errorf("expected 2 sentences, got %d", len(summary))
				}
			}
		})
	}
}
