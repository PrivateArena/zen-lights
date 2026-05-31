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
	"github.com/zen-lights/zen-lights/internal/paint"
	"github.com/zen-lights/zen-lights/internal/preview"
	"github.com/zen-lights/zen-lights/internal/summarize"
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
	case "server":
		runServer(os.Args[2:])
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
	fmt.Println("  server      Start a persistent unified HTTP API server (OCR, translation, summarization, paint)")
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

func runServer(args []string) {
	fs := flag.NewFlagSet("server", flag.ExitOnError)
	addr := fs.String("addr", "localhost:8080", "address for the unified server")
	configPath := fs.String("config", "config.json", "path to the configuration profiles")
	defaultModel := fs.String("default-model", "ch", "default OCR model/language profile to use")
	fs.Parse(args)

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
	defer transManager.Close()

	// Initialize summarize manager
	sumManager := summarize.NewManager(summarize.Config{})
	if err := sumManager.LoadConfig(*configPath); err != nil {
		log.Printf("Warning: failed to load summarize config from %s: %v", *configPath, err)
	} else {
		log.Printf("Loaded summarize profiles from %s", *configPath)
	}

	// Initialize paint manager
	paintManager := paint.NewManager(paint.DefaultConfig)
	if err := paintManager.LoadConfig(*configPath); err != nil {
		log.Printf("Warning: failed to load paint config from %s: %v", *configPath, err)
	} else {
		log.Printf("Loaded paint profiles from %s", *configPath)
	}
	defer paintManager.Close()

	srv := server.New(*addr, manager, *defaultModel, transManager, sumManager, paintManager)
	if err := srv.Start(); err != nil {
		log.Fatal("server:", err)
	}
}

func totalKills(r *pipeline.Result) int {
	n := 0
	for _, s := range r.Segments {
		n += s.TotalKills()
	}
	return n
}
