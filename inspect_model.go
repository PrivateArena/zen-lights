package main

import (
	"fmt"
	"os"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	libPath := "/media/jang/home/Deve/zen-tts/piper/libonnxruntime.so.1.24.2"
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		fmt.Printf("Init error: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	modelPath := "models/japan_PP-OCRv4_rec_infer.onnx"
	session, err := ort.NewAdvancedSession(modelPath, nil, nil, nil)
	if err != nil {
		fmt.Printf("Session error: %v\n", err)
		return
	}
	defer session.Destroy()

	fmt.Println("Inputs:")
	for i := 0; i < session.GetInputCount(); i++ {
		name, _ := session.GetInputName(i)
		fmt.Printf("  [%d] %s\n", i, name)
	}

	fmt.Println("Outputs:")
	for i := 0; i < session.GetOutputCount(); i++ {
		name, _ := session.GetOutputName(i)
		fmt.Printf("  [%d] %s\n", i, name)
	}
}
