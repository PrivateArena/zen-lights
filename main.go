// Command zenlights extracts and merges highlight clips from esport/sport VODs.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zen-lights/zen-lights/internal/ocr"
	"github.com/zen-lights/zen-lights/internal/ocr/server"
	"github.com/zen-lights/zen-lights/internal/preview"
	"github.com/zen-lights/zen-lights/internal/translate"
	"github.com/zen-lights/zen-lights/pkg/game"
	"github.com/zen-lights/zen-lights/pkg/pipeline"

	// Register game detectors — add new games here
	_ "github.com/zen-lights/zen-lights/pkg/game/cs2"
	_ "github.com/zen-lights/zen-lights/pkg/game/dota2"
	_ "github.com/zen-lights/zen-lights/pkg/game/lol"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("zenlights: ")

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "detect":
		runDetect(os.Args[2:])
	case "ocr-server":
		runOCRServer(os.Args[2:])
	case "help", "-h", "--help":
		printUsage()
	default:
		// Fallback for backward compatibility (treat as detect)
		runDetect(os.Args[1:])
	}
}

func printUsage() {
	fmt.Println("Usage: zenlights <command> [options]")
	fmt.Println("\nCommands:")
	fmt.Println("  detect      Run the highlight extraction pipeline (default)")
	fmt.Println("  ocr-server  Start a persistent HTTP OCR server with multi-language support")
	fmt.Println("\nRun 'zenlights <command> -h' for more info on a command.")
}

func runDetect(args []string) {
	fs := flag.NewFlagSet("detect", flag.ExitOnError)
	var (
		inputPath    = fs.String("input", "", "path to the input video file (required)")
		outputPath   = fs.String("output", "highlights.mp4", "path for the merged output video")
		gameName     = fs.String("game", "", fmt.Sprintf("game to detect — one of: %s", strings.Join(game.Available(), ", ")))
		doPreview    = fs.Bool("preview", false, "open an HTTP preview server after processing")
		previewAddr  = fs.String("addr", "localhost:8765", "address for the preview server")
		verbose      = fs.Bool("v", false, "verbose logging")
		dumpFrames   = fs.String("dump-frames", "", "directory to write preprocessed OCR frames (for ROI/threshold debugging)")
		maxScoreJump = fs.Int("max-score-jump", 5, "maximum plausible kill-score increase per sample frame (reject OCR noise above this)")
	)
	fs.Parse(args)

	// ── Validate flags ────────────────────────────────────────────────────────
	var errs []string
	if *inputPath == "" {
		errs = append(errs, "-input is required")
	}
	if *gameName == "" {
		errs = append(errs, fmt.Sprintf("-game is required (available: %s)", strings.Join(game.Available(), ", ")))
	}
	if len(errs) > 0 {
		for _, e := range errs {
			log.Println("error:", e)
		}
		os.Exit(1)
	}

	detector, err := game.Get(*gameName)
	if err != nil {
		log.Fatal(err)
	}

	// ── Run pipeline ──────────────────────────────────────────────────────────
	result, err := pipeline.Run(pipeline.Config{
		InputPath:    *inputPath,
		OutputPath:   *outputPath,
		Game:         detector,
		Verbose:      *verbose,
		DumpFrameDir: *dumpFrames,
		MaxScoreJump: *maxScoreJump,
	})
	if err != nil {
		log.Fatal(err)
	}

	// ── Summary ───────────────────────────────────────────────────────────────
	fmt.Printf("\n✅ Highlights written to: %s\n", result.OutputPath)
	fmt.Printf("   %d segment(s), %d total kills\n\n", len(result.Segments), totalKills(result))
	for i, seg := range result.Segments {
		fmt.Printf("   [%2d] %s\n", i+1, seg)
	}
	fmt.Println()

	// ── Optional preview server ───────────────────────────────────────────────
	if *doPreview {
		if err := preview.Serve(*previewAddr, result.OutputPath, result.Segments); err != nil {
			log.Fatal("preview server:", err)
		}
	}
}

func runOCRServer(args []string) {
	fs := flag.NewFlagSet("ocr-server", flag.ExitOnError)
	addr := fs.String("addr", "localhost:8080", "address for the OCR server")
	configPath := fs.String("config", "config.json", "path to the language profiles config file")
	defaultModel := fs.String("default-model", "ch", "default OCR model/language to use if not specified in API")
	fs.Parse(args)

	// Robustly handle mixed positional args (e.g. "ocr-server 8765 -default-model en")
	positionalArgs := fs.Args()
	if len(positionalArgs) > 0 && !strings.HasPrefix(positionalArgs[0], "-") {
		val := positionalArgs[0]
		if isPurePort(val) {
			*addr = "localhost:" + val
		} else {
			*addr = val
		}
		positionalArgs = positionalArgs[1:]
	}

	for i := 0; i < len(positionalArgs); i++ {
		arg := positionalArgs[i]
		if arg == "-default-model" || arg == "--default-model" {
			if i+1 < len(positionalArgs) {
				*defaultModel = positionalArgs[i+1]
				i++
			}
		} else if arg == "-config" || arg == "--config" {
			if i+1 < len(positionalArgs) {
				*configPath = positionalArgs[i+1]
				i++
			}
		} else if arg == "-addr" || arg == "--addr" {
			if i+1 < len(positionalArgs) {
				*addr = positionalArgs[i+1]
				i++
			}
		}
	}

	manager := ocr.NewManager(ocr.DefaultOptions())

	// Load language profiles from config file
	if err := manager.LoadConfig(*configPath); err != nil {
		log.Printf("Warning: failed to load config from %s: %v", *configPath, err)
		log.Println("Proceeding with empty language registry (you may need to register languages via API later if implemented)")
	} else {
		log.Printf("Loaded language profiles from %s", *configPath)
	}

	// Initialize translation manager
	transManager := translate.NewManager(translate.DefaultConfig())
	if err := transManager.LoadConfig(*configPath); err != nil {
		log.Printf("Warning: failed to load translation config from %s: %v", *configPath, err)
	} else {
		log.Printf("Loaded translation profiles from %s", *configPath)
	}

	finalDefaultModel := *defaultModel
	userSetDefaultModel := false
	for _, arg := range args {
		if strings.HasPrefix(arg, "-default-model") || strings.HasPrefix(arg, "--default-model") {
			userSetDefaultModel = true
			break
		}
	}
	if !userSetDefaultModel && manager.DefaultModel() != "" {
		finalDefaultModel = manager.DefaultModel()
	}

	srv := server.New(*addr, manager, finalDefaultModel, transManager)
	if err := srv.Start(); err != nil {
		log.Fatal("ocr-server:", err)
	}
}

func isPurePort(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return true
}

func totalKills(r *pipeline.Result) int {
	n := 0
	for _, s := range r.Segments {
		n += s.TotalKills()
	}
	return n
}
