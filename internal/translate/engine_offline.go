package translate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortInitOnce sync.Once
	ortInitErr  error
)

// initORT sets the shared library path and initializes the environment.
func initORT(sharedLibPath string) error {
	ortInitOnce.Do(func() {
		libPath := sharedLibPath
		if libPath == "" {
			libPath = os.Getenv("ORT_SHARED_LIB_PATH")
		}
		if libPath == "" {
			candidates := []string{
				"/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2",
				"./piper/libonnxruntime.so.1.24.2",
				"../piper/libonnxruntime.so.1.24.2",
				"../../piper/libonnxruntime.so.1.24.2",
				"./models/libonnxruntime.so",
			}
			for _, c := range candidates {
				abs, err := filepath.Abs(c)
				if err == nil {
					if _, err := os.Stat(abs); err == nil {
						libPath = abs
						break
					}
				}
			}
		}
		if libPath == "" {
			libPath = "libonnxruntime.so"
		}

		ort.SetSharedLibraryPath(libPath)
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// OfflineEngine implements Engine using offline ONNX models.
type OfflineEngine struct {
	encoderSession *ort.DynamicAdvancedSession
	decoderSession *ort.DynamicAdvancedSession
	tokenizer      *Tokenizer
	maxTokens      int
	padTokenID     int64
	eosTokenID     int64
}

// NewOfflineEngine creates a new offline translation engine from a profile.
func NewOfflineEngine(profile TranslationProfile, sharedLibPath string, maxTokens int) (*OfflineEngine, error) {
	if err := initORT(sharedLibPath); err != nil {
		return nil, fmt.Errorf("init onnxruntime: %w", err)
	}

	tok, err := NewTokenizer(
		profile.SourceSPM,
		profile.TargetSPM,
		profile.VocabPath,
		profile.PadTokenID,
		profile.EosTokenID,
	)
	if err != nil {
		return nil, fmt.Errorf("init tokenizer: %w", err)
	}

	encoderSession, err := ort.NewDynamicAdvancedSession(
		profile.EncoderPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"last_hidden_state"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create encoder session: %w", err)
	}

	decoderSession, err := ort.NewDynamicAdvancedSession(
		profile.DecoderPath,
		[]string{"input_ids", "encoder_hidden_states", "encoder_attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		encoderSession.Destroy()
		return nil, fmt.Errorf("create decoder session: %w", err)
	}

	if maxTokens <= 0 {
		maxTokens = 128
	}

	return &OfflineEngine{
		encoderSession: encoderSession,
		decoderSession: decoderSession,
		tokenizer:      tok,
		maxTokens:      maxTokens,
		padTokenID:     profile.PadTokenID,
		eosTokenID:     profile.EosTokenID,
	}, nil
}

// Translate translates the given text using offline ONNX models.
func (e *OfflineEngine) Translate(ctx context.Context, text, srcLang, tgtLang string) (string, error) {
	if text == "" {
		return "", nil
	}

	inputIDs := e.tokenizer.Encode(text)

	outputIDs, err := greedyDecode(
		ctx,
		inputIDs,
		e.encoderSession,
		e.decoderSession,
		e.maxTokens,
		e.padTokenID,
		e.eosTokenID,
	)
	if err != nil {
		return "", fmt.Errorf("greedy decode: %w", err)
	}

	return e.tokenizer.Decode(outputIDs), nil
}

// Close destroys the underlying ONNX sessions.
func (e *OfflineEngine) Close() error {
	var errs []error
	if e.encoderSession != nil {
		if err := e.encoderSession.Destroy(); err != nil {
			errs = append(errs, err)
		}
		e.encoderSession = nil
	}
	if e.decoderSession != nil {
		if err := e.decoderSession.Destroy(); err != nil {
			errs = append(errs, err)
		}
		e.decoderSession = nil
	}
	if len(errs) > 0 {
		return fmt.Errorf("errors closing sessions: %v", errs)
	}
	return nil
}
