// Package highlight defines the Segment type and groups raw kill events
// into highlight clips, merging nearby kills into single segments.
package highlight

import (
	"fmt"
	"time"

	"github.com/zen-lights/zen-lights/pkg/game"
)

// Segment is a contiguous highlight clip that covers one or more kill events.
// Start and End already include the PreBuffer and PostBuffer from the Detector.
type Segment struct {
	Events []game.KillEvent
	Start  time.Duration
	End    time.Duration
}

// TotalKills returns the sum of all kill deltas across the segment's events.
func (s Segment) TotalKills() int {
	n := 0
	for _, e := range s.Events {
		n += e.Delta()
	}
	return n
}

// Duration returns the length of the highlight clip.
func (s Segment) Duration() time.Duration { return s.End - s.Start }

// String returns a human-readable summary of the segment.
func (s Segment) String() string {
	return fmt.Sprintf("[%s → %s] %d kills over %d event(s)",
		fmtDur(s.Start), fmtDur(s.End), s.TotalKills(), len(s.Events))
}

func fmtDur(d time.Duration) string {
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	sec := d.Seconds() - float64(h*3600) - float64(m*60)
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%05.2f", h, m, sec)
	}
	return fmt.Sprintf("%d:%05.2f", m, sec)
}

// GroupEvents groups a flat list of KillEvents into highlight Segments.
//
// Events that occur within mergeWindow of each other are merged into one
// segment (capturing multi-kill chains). Each segment is then padded with
// preBuf before the first kill and postBuf after the last kill.
func GroupEvents(
	events []game.KillEvent,
	mergeWindow, preBuf, postBuf time.Duration,
	videoDuration time.Duration,
) []Segment {
	if len(events) == 0 {
		return nil
	}

	var segments []Segment
	cur := Segment{Events: []game.KillEvent{events[0]}}

	for _, ev := range events[1:] {
		lastAt := cur.Events[len(cur.Events)-1].At
		if ev.At-lastAt <= mergeWindow {
			// Still within merge window — extend the current segment
			cur.Events = append(cur.Events, ev)
		} else {
			segments = append(segments, finalize(cur, preBuf, postBuf, videoDuration))
			cur = Segment{Events: []game.KillEvent{ev}}
		}
	}
	segments = append(segments, finalize(cur, preBuf, postBuf, videoDuration))

	return segments
}

func finalize(s Segment, preBuf, postBuf, videoDuration time.Duration) Segment {
	s.Start = s.Events[0].At - preBuf
	if s.Start < 0 {
		s.Start = 0
	}
	s.End = s.Events[len(s.Events)-1].At + postBuf
	if videoDuration > 0 && s.End > videoDuration {
		s.End = videoDuration
	}
	return s
}
