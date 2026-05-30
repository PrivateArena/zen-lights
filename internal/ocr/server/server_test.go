package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/zen-lights/zen-lights/internal/ocr"
)

func TestDefaultModelEndpoints(t *testing.T) {
	// Create manager with empty options
	manager := ocr.NewManager(ocr.DefaultOptions())

	// Register dummy languages
	manager.RegisterLanguage(ocr.LanguageProfile{ID: "ch"})
	manager.RegisterLanguage(ocr.LanguageProfile{ID: "en"})
	manager.RegisterLanguage(ocr.LanguageProfile{ID: "ja"})

	// Create server
	srv := New(":0", manager, "ch")

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
