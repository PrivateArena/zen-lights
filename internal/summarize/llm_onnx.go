package summarize

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortOnce    sync.Once
	ortInitErr error

	sessionOnce sync.Once
	sessionErr  error
	session     *ort.DynamicAdvancedSession
	sessionMu   sync.Mutex

	// Model configurations set at initialization
	tokenizer *SimpleTokenizer
	numLayers = 18
	vocabSize = int64(262144)
)

// InitEnvironment ensures the ONNX runtime library is loaded and the environment is ready.
func InitEnvironment(sharedLibPath string) error {
	ortOnce.Do(func() {
		if sharedLibPath != "" {
			ort.SetSharedLibraryPath(sharedLibPath)
		}
		ortInitErr = ort.InitializeEnvironment()
	})
	return ortInitErr
}

// InitONNX initializes the ONNX session once with specified config parameters.
func InitONNX(sharedLibPath, onnxModelPath string, tok *SimpleTokenizer, layers int, vSize int64) error {
	if err := InitEnvironment(sharedLibPath); err != nil {
		return err
	}

	sessionOnce.Do(func() {
		tokenizer = tok
		if layers > 0 {
			numLayers = layers
		}
		if vSize > 0 {
			vocabSize = vSize
		}

		inputNames := []string{"input_ids", "attention_mask"}
		outputNames := []string{"logits"}
		for i := 0; i < numLayers; i++ {
			inputNames = append(inputNames, fmt.Sprintf("past_key_values.%d.key", i))
			inputNames = append(inputNames, fmt.Sprintf("past_key_values.%d.value", i))
			outputNames = append(outputNames, fmt.Sprintf("present.%d.key", i))
			outputNames = append(outputNames, fmt.Sprintf("present.%d.value", i))
		}

		opts, err := ort.NewSessionOptions()
		if err != nil {
			sessionErr = err
			return
		}
		_ = opts.SetIntraOpNumThreads(4)

		session, sessionErr = ort.NewDynamicAdvancedSession(onnxModelPath, inputNames, outputNames, opts)
	})
	return sessionErr
}

// SummarizeWithLLM takes raw text, formats it as a Gemma-3 user-prompt requesting summarization, and generates output tokens.
func SummarizeWithLLM(text string, maxTokens int) (string, error) {
	prompt := fmt.Sprintf("<bos><start_of_turn>user\nSummarize the following text concisely:\n\n%s<end_of_turn>\n<start_of_turn>model\n", text)
	return runGreedyGeneration(prompt, maxTokens)
}

func runGreedyGeneration(prompt string, maxTokens int) (string, error) {
	sessionMu.Lock()
	defer sessionMu.Unlock()

	if session == nil {
		return "", fmt.Errorf("ONNX session not initialized")
	}
	if tokenizer == nil {
		return "", fmt.Errorf("tokenizer not initialized")
	}

	inputIDs, err := tokenizer.Encode(prompt)
	if err != nil {
		return "", fmt.Errorf("encode prompt error: %w", err)
	}

	// Initial KV Cache: Empty tensors for each layer (batch=1, heads=1, len=0, dim=256)
	kvValues := make([]ort.Value, numLayers*2)
	for i := range kvValues {
		t, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, 0, 256))
		if err != nil {
			// Cleanup previously allocated tensors on failure
			for j := 0; j < i; j++ {
				if kvValues[j] != nil {
					kvValues[j].Destroy()
				}
			}
			return "", fmt.Errorf("failed to create empty KV cache tensor: %w", err)
		}
		kvValues[i] = t
	}
	defer func() {
		for _, v := range kvValues {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	var generatedTokens []int64
	currentInput := inputIDs

	for step := 0; step < maxTokens; step++ {
		seqLen := int64(len(currentInput))
		totalLen := int64(len(inputIDs)) + int64(len(generatedTokens))

		// 1. Prepare Inputs
		idTensor, err := ort.NewTensor(ort.NewShape(1, seqLen), currentInput)
		if err != nil {
			return "", fmt.Errorf("create input tensor: %w", err)
		}
		mask := make([]int64, totalLen)
		for i := range mask {
			mask[i] = 1
		}
		maskTensor, err := ort.NewTensor(ort.NewShape(1, totalLen), mask)
		if err != nil {
			idTensor.Destroy()
			return "", fmt.Errorf("create mask tensor: %w", err)
		}

		inputs := make([]ort.Value, 2+len(kvValues))
		inputs[0] = idTensor
		inputs[1] = maskTensor
		for i, v := range kvValues {
			inputs[2+i] = v
		}

		// 2. Prepare Outputs
		outputs := make([]ort.Value, 1+len(kvValues))

		// 3. Run Inference
		err = session.Run(inputs, outputs)
		idTensor.Destroy()
		maskTensor.Destroy()
		if err != nil {
			return "", fmt.Errorf("step %d run error: %w", step, err)
		}

		// 4. Process Logits (Output 0 is logits)
		logitsTensor, ok := outputs[0].(*ort.Tensor[float32])
		if !ok {
			return "", fmt.Errorf("invalid logits output type")
		}
		logits := logitsTensor.GetData()
		lastPos := (seqLen - 1) * vocabSize
		if lastPos+vocabSize > int64(len(logits)) {
			logitsTensor.Destroy()
			return "", fmt.Errorf("logits size mismatch: expected last pos range %d..%d, got size %d", lastPos, lastPos+vocabSize, len(logits))
		}
		nextToken := argmax(logits[lastPos : lastPos+vocabSize])

		// Update KV Cache with present ones (Outputs 1 to N)
		for i := range kvValues {
			if kvValues[i] != nil {
				kvValues[i].Destroy()
			}
			kvValues[i] = outputs[i+1]
		}
		logitsTensor.Destroy()

		// 1 = EOS, 106 = model turn end (<end_of_turn>)
		if nextToken == 1 || nextToken == 106 {
			break
		}
		generatedTokens = append(generatedTokens, int64(nextToken))
		currentInput = []int64{int64(nextToken)}
	}

	decoded, err := tokenizer.Decode(generatedTokens)
	if err != nil {
		return "", fmt.Errorf("decode error: %w", err)
	}
	return decoded, nil
}

func argmax(data []float32) int {
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
