package translate

import (
	"context"
	"errors"
)

// Mode defines how translation should be executed.
type Mode string

const (
	ModeOnline  Mode = "online"  // Direct online (Google)
	ModeOffline Mode = "offline" // Direct offline (ONNX)
	ModeAuto    Mode = "auto"    // Try offline, fallback to online
)

// Engine is the common interface implemented by translation backends.
type Engine interface {
	Translate(ctx context.Context, text, srcLang, tgtLang string) (string, error)
	Close() error
}

var (
	ErrUnsupportedLanguagePair = errors.New("unsupported language pair")
	ErrOfflineFailed           = errors.New("offline translation failed")
)

// TranslationProfile defines the model path configuration for a specific language pair.
type TranslationProfile struct {
	ID          string `json:"id"`           // e.g., "ja-en", "ko-en", "zh-en"
	EncoderPath string `json:"encoder_path"` // Path to encoder_model.onnx
	DecoderPath string `json:"decoder_path"` // Path to decoder_model.onnx (or decoder_model_merged.onnx)
	SourceSPM   string `json:"source_spm"`   // Path to source.spm
	TargetSPM   string `json:"target_spm"`   // Path to target.spm
	VocabPath   string `json:"vocab_path"`   // Path to vocab.json
	EosTokenID  int64  `json:"eos_token_id"` // Typically 0 for Helsinki models
	PadTokenID  int64  `json:"pad_token_id"` // Typically 65000 for Helsinki models
}

// Config represents the translation configuration.
type Config struct {
	Mode      Mode                 `json:"mode"`       // "online", "offline", "auto"
	MaxTokens int                  `json:"max_tokens"` // Max tokens to generate in offline mode (default 128)
	Profiles  []TranslationProfile `json:"profiles"`
}

// DefaultConfig returns a sane default translation configuration.
func DefaultConfig() Config {
	return Config{
		Mode:      ModeAuto,
		MaxTokens: 128,
	}
}
