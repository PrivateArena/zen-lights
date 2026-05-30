package translate

import (
	"context"
	"errors"

	ort "github.com/yalue/onnxruntime_go"
)

// greedyDecode runs autoregressive generation loop using the encoder and decoder sessions.
func greedyDecode(
	ctx context.Context,
	inputIDs []int64,
	encoderSession *ort.DynamicAdvancedSession,
	decoderSession *ort.DynamicAdvancedSession,
	maxTokens int,
	padTokenID, eosTokenID int64,
) ([]int64, error) {
	if len(inputIDs) == 0 {
		return nil, nil
	}

	srcSeqLen := int64(len(inputIDs))
	attentionMask := make([]int64, srcSeqLen)
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	encoderInputShape := ort.NewShape(1, srcSeqLen)
	inputIDsTensor, err := ort.NewTensor(encoderInputShape, inputIDs)
	if err != nil {
		return nil, err
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(encoderInputShape, attentionMask)
	if err != nil {
		return nil, err
	}
	defer attentionMaskTensor.Destroy()

	encoderOutputs := []ort.Value{nil}
	err = encoderSession.Run([]ort.Value{inputIDsTensor, attentionMaskTensor}, encoderOutputs)
	if err != nil {
		return nil, err
	}

	hiddenStateTensor, ok := encoderOutputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, errors.New("failed casting last_hidden_state output")
	}
	defer hiddenStateTensor.Destroy()

	decoderIDs := []int64{padTokenID}

	for step := 0; step < maxTokens; step++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		decSeqLen := int64(len(decoderIDs))
		decInputShape := ort.NewShape(1, decSeqLen)
		decInputTensor, err := ort.NewTensor(decInputShape, decoderIDs)
		if err != nil {
			return nil, err
		}

		decoderOutputs := []ort.Value{nil}
		err = decoderSession.Run([]ort.Value{
			decInputTensor,
			hiddenStateTensor,
			attentionMaskTensor,
		}, decoderOutputs)
		decInputTensor.Destroy()

		if err != nil {
			return nil, err
		}

		logitsTensor, ok := decoderOutputs[0].(*ort.Tensor[float32])
		if !ok {
			return nil, errors.New("failed casting logits output")
		}

		logitsShape := logitsTensor.GetShape()
		vocabSize := int(logitsShape[2])
		logitsData := logitsTensor.GetData()
		logitsTensor.Destroy()

		lastTokenOffset := (int(decSeqLen) - 1) * vocabSize
		lastTokenLogits := logitsData[lastTokenOffset : lastTokenOffset+vocabSize]

		var nextTokenID int64
		var maxVal float32 = -1e9
		for id, val := range lastTokenLogits {
			if val > maxVal {
				maxVal = val
				nextTokenID = int64(id)
			}
		}

		if nextTokenID == eosTokenID {
			break
		}

		decoderIDs = append(decoderIDs, nextTokenID)
	}

	if len(decoderIDs) > 1 {
		return decoderIDs[1:], nil
	}
	return nil, nil
}
