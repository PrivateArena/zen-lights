package pipeline

import (
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// dumpFramePNG writes a preprocessed OCR frame PNG to dir for debugging.
// Files are named by frame index and timestamp so they can be reviewed in order.
// Non-fatal: errors are silently ignored so debugging doesn't break the pipeline.
func dumpFramePNG(dir string, frameIdx int, ts time.Duration, pngBytes []byte) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	ms := ts.Milliseconds()
	name := fmt.Sprintf("frame_%06d_%07dms.png", frameIdx, ms)
	path := filepath.Join(dir, name)
	return os.WriteFile(path, pngBytes, 0o644)
}
