package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zen-lights/zen-lights/internal/ocr"
	"github.com/zen-lights/zen-lights/internal/summarize"
	"github.com/zen-lights/zen-lights/internal/translate"
)

func TestDefaultModelEndpoints(t *testing.T) {
	// Create manager with empty options
	manager := ocr.NewManager(ocr.DefaultOptions())

	// Register dummy languages
	manager.RegisterLanguage(ocr.LanguageProfile{ID: "ch"})
	manager.RegisterLanguage(ocr.LanguageProfile{ID: "en"})
	manager.RegisterLanguage(ocr.LanguageProfile{ID: "ja"})

	// Create translate manager with empty config
	transManager := translate.NewManager(translate.Config{Mode: translate.ModeOnline})

	// Create server
	srv := New(":0", manager, "ch", transManager, nil)

	// Create test HTTP recorder for GET /default-model
	req, err := http.NewRequest("GET", "/default-model", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	srv.handleDefaultModel(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rr.Code)
	}

	var getResp map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &getResp); err != nil {
		t.Fatal(err)
	}
	if getResp["default_model"] != "ch" {
		t.Errorf("expected default model 'ch', got '%s'", getResp["default_model"])
	}

	// Test POST /default-model?model=en
	req, err = http.NewRequest("POST", "/default-model?model=en", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr = httptest.NewRecorder()
	srv.handleDefaultModel(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", rr.Code)
	}

	var postResp map[string]string
	if err := json.Unmarshal(rr.Body.Bytes(), &postResp); err != nil {
		t.Fatal(err)
	}
	if postResp["default_model"] != "en" {
		t.Errorf("expected default model to be updated to 'en', got '%s'", postResp["default_model"])
	}
	if srv.defaultModel != "en" {
		t.Errorf("expected server field to be 'en', got '%s'", srv.defaultModel)
	}

	// Test POST with invalid/non-existent model
	req, err = http.NewRequest("POST", "/default-model?model=fr", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr = httptest.NewRecorder()
	srv.handleDefaultModel(rr, req)

	if rr.Code != http.StatusNotFound {
		t.Errorf("expected status 404 for missing model, got %d", rr.Code)
	}
}

func TestTranslateEndpoint(t *testing.T) {
	manager := ocr.NewManager(ocr.DefaultOptions())
	transManager := translate.NewManager(translate.Config{Mode: translate.ModeOnline})
	srv := New(":0", manager, "ch", transManager, nil)

	// Since we mock the actual online translation, let's write a simple validation 
	// for the missing query/body parameter error.
	req, err := http.NewRequest("GET", "/translate", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	srv.handleTranslate(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected status 400 for empty request, got %d", rr.Code)
	}
}

func TestSummarizeEndpoint(t *testing.T) {
	manager := ocr.NewManager(ocr.DefaultOptions())
	transManager := translate.NewManager(translate.Config{Mode: translate.ModeOnline})
	sumManager := summarize.NewManager(summarize.Config{
		Algorithm: summarize.AlgoTextRank,
	})

	srv := New(":0", manager, "ch", transManager, sumManager)

	// 1. Test missing 'text' parameter
	req, err := http.NewRequest("GET", "/summarize", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	srv.handleSummarize(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Errorf("expected status 400 for empty text, got %d", rr.Code)
	}

	// 2. Test successful summarization with extractive engine
	reqText := "First sentence is interesting. Second sentence is also quite interesting. Third sentence is not so interesting."
	req, err = http.NewRequest("GET", "/summarize?text="+reqText+"&count=2", nil)
	if err != nil {
		t.Fatal(err)
	}
	rr = httptest.NewRecorder()
	srv.handleSummarize(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d. Body: %s", rr.Code, rr.Body.String())
	}

	var resp summarizeResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}

	if len(resp.Summary) != 2 {
		t.Errorf("expected 2 summary sentences, got %d", len(resp.Summary))
	}
}
