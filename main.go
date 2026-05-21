// Command zenlights extracts and merges highlight clips from MOBA gameplay videos.
//
// Usage:
//
//	zenlights -game lol -input match.mp4 -output highlights.mp4 [-preview] [-v]
//
// Supported games: lol, dota2
// (add more by importing their detector packages below)
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zen-lights/zen-lights/internal/preview"
	"github.com/zen-lights/zen-lights/pkg/game"
	"github.com/zen-lights/zen-lights/pkg/pipeline"

	// Register game detectors — add new games here
	_ "github.com/zen-lights/zen-lights/pkg/game/dota2"
	_ "github.com/zen-lights/zen-lights/pkg/game/lol"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("zenlights: ")

	var (
		inputPath   = flag.String("input", "", "path to the input video file (required)")
		outputPath  = flag.String("output", "highlights.mp4", "path for the merged output video")
		gameName    = flag.String("game", "", fmt.Sprintf("game to detect (required) — one of: %s", strings.Join(game.Available(), ", ")))
		doPreview   = flag.Bool("preview", false, "open an HTTP preview server after processing")
		previewAddr = flag.String("addr", "localhost:8765", "address for the preview server")
		verbose     = flag.Bool("v", false, "verbose logging")
	)
	flag.Parse()

	// ── Validate flags ───────────────────────────────────────────────────────
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
		fmt.Fprintln(os.Stderr)
		flag.Usage()
		os.Exit(1)
	}

	detector, err := game.Get(*gameName)
	if err != nil {
		log.Fatal(err)
	}

	// ── Run pipeline ─────────────────────────────────────────────────────────
	result, err := pipeline.Run(pipeline.Config{
		InputPath:  *inputPath,
		OutputPath: *outputPath,
		Game:       detector,
		Verbose:    *verbose,
	})
	if err != nil {
		log.Fatal(err)
	}

	// ── Print summary ─────────────────────────────────────────────────────────
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

func totalKills(r *pipeline.Result) int {
	n := 0
	for _, s := range r.Segments {
		n += s.TotalKills()
	}
	return n
}
