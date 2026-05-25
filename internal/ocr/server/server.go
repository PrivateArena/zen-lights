package server

import (
	"encoding/json"
	"fmt"
	"image"
	"log"
	"net"
	"net/http"

	_ "image/jpeg"
	_ "image/png"

	"github.com/zen-lights/zen-lights/internal/ocr"
)

// Server provides a persistent HTTP API for multi-language OCR.
type Server struct {
	manager *ocr.Manager
	addr    string
}

// New creates a new OCR server.
func New(addr string, manager *ocr.Manager) *Server {
	return &Server{
		addr:    addr,
		manager: manager,
	}
}

// Start runs the HTTP server.
func (s *Server) Start() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/recognize", s.handleRecognize)
	mux.HandleFunc("/status", s.handleStatus)

	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.addr, err)
	}

	log.Printf("🤖 OCR Server listening on http://%s", ln.Addr())
	return http.Serve(ln, mux)
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

type recognizeResponse struct {
	Results []ocr.TextResult `json:"results"`
	Error   string           `json:"error,omitempty"`
}

func (s *Server) handleRecognize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// 1. Get language ID
	langID := r.URL.Query().Get("lang")
	if langID == "" {
		langID = "ch" // Default to Chinese
	}

	// 2. Parse image from request body (multipart or raw)
	var img image.Image
	file, _, err := r.FormFile("image")
	if err == nil {
		defer file.Close()
		img, _, err = image.Decode(file)
	} else {
		// Try reading raw body if multipart fails
		img, _, err = image.Decode(r.Body)
	}

	if err != nil {
		s.jsonError(w, fmt.Sprintf("Failed to decode image: %v", err), http.StatusBadRequest)
		return
	}

	// 3. Get OCR client for the language
	client, err := s.manager.GetClient(langID)
	if err != nil {
		s.jsonError(w, fmt.Sprintf("Language %q not available: %v", langID, err), http.StatusNotFound)
		return
	}

	// 4. Run OCR
	// Convert image to RGB24 frameData for Client.ReadFullFrame
	bounds := img.Bounds()
	wImg, hImg := bounds.Dx(), bounds.Dy()
	frameData := make([]byte, wImg*hImg*3)
	for y := 0; y < hImg; y++ {
		for x := 0; x < wImg; x++ {
			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			idx := (y*wImg + x) * 3
			frameData[idx] = uint8(r >> 8)
			frameData[idx+1] = uint8(g >> 8)
			frameData[idx+2] = uint8(b >> 8)
		}
	}

	results, err := client.ReadFullFrame(frameData, wImg, hImg)
	if err != nil {
		s.jsonError(w, fmt.Sprintf("OCR processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// 5. Respond
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(recognizeResponse{Results: results})
}

func (s *Server) jsonError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(recognizeResponse{Error: msg})
}
