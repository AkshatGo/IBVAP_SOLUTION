# IBVAP Demo Video Recording Guide

## 🎬 Overview

This guide walks you through creating a professional demo video for your SIH 2026 presentation. The demo should be 3-4 minutes long and showcase the key features of IBVAP.

---

## 📋 Prerequisites

### Software Requirements
- Python 3.8+
- OpenCV (`pip install opencv-python`)
- Optional: pyautogui or mss for screen recording

### Hardware Requirements
- Screen resolution: 1920x1080 (Full HD) recommended
- Stable mouse and keyboard
- Quiet environment for narration

---

## 🎥 Recording Options

### Option 1: Automated Demo (Recommended for First-Time Users)

Run the automated demo generator:

```bash
cd SIH2026
python scripts/record_demo.py --scene all
```

This will generate:
- `demo_videos/01_title.mp4` - Title screen
- `demo_videos/02_virtual_fence.mp4` - Virtual fence demo
- `demo_videos/03_anpr_consensus.mp4` - ANPR demo
- `demo_videos/04_signal_loss.mp4` - Signal loss demo
- `demo_videos/05_hash_chain.mp4` - Hash chain demo

### Option 2: Screen Recording (For Live Demo)

```bash
# Interactive recording (stops with Ctrl+C)
python scripts/screen_recorder.py --mode interactive --output live_demo.mp4

# Timed recording (60 seconds)
python scripts/screen_recorder.py --mode timed --duration 60 --output timed_demo.mp4

# Full presentation recording
python scripts/screen_recorder.py --mode demo
```

### Option 3: Manual Recording with OBS Studio

1. Download and install [OBS Studio](https://obsproject.com/)
2. Set up scene with your application window
3. Start recording before each demo segment
4. Edit together in post

---

## 🎬 Demo Script & Timing

### Scene 1: Title Screen (0:00 - 0:05)
**Duration:** 5 seconds

**What to show:**
- IBVAP logo/title
- One-liner: "Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert..."

**Narration:**
> "IBVAP - Intelligent Border Video Analytics Platform. Every AI-CCTV platform on the market assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three."

---

### Scene 2: Architecture Overview (0:05 - 0:35)
**Duration:** 30 seconds

**What to show:**
- Three-tier architecture diagram
- Cost callouts ($250 Tier-1, $50 Tier-2)

**Narration:**
> "We designed a three-tier system. Tier 1: Jetson Orin Nano at $250 for full AI at the camera site. Tier 2: $20-30 microcontroller for remote cameras. Tier 3: Command Center that aggregates everything. Only metadata travels upstream, never video."

---

### Scene 3: Virtual Fence Demo (0:35 - 1:35)
**Duration:** 60 seconds

**What to show:**
1. Open dashboard
2. Draw virtual fence polygon on camera view
3. Show person/vehicle approaching
4. Alert fires when crossing fence
5. Show alert with explanation field

**Narration:**
> "Watch this. I'm drawing a virtual fence on the camera view. When someone crosses this boundary, the system fires an alert in under 3 seconds. Notice the explanation field - the operator sees WHY it fired, not just a bounding box."

**Key points to highlight:**
- Drawing the fence polygon
- Real-time tracking
- Alert with explanation
- <3 second latency

---

### Scene 4: ANPR Demo (1:35 - 2:35)
**Duration:** 60 seconds

**What to show:**
1. Vehicle approaching camera
2. Single-frame OCR result (noisy)
3. Multi-frame consensus voting
4. Final consensus result

**Narration:**
> "Single-frame OCR on border plates fails constantly. Watch what happens with our multi-frame approach. Frame 1: BR12AB3456. Frame 2: BR12AB3458. Frame 3: BR12AB3456. The system takes majority vote: BR12AB3456. This is genuinely better than single-frame in front of judges."

**Key points to highlight:**
- Single-frame failure
- Multi-frame voting
- Consensus result
- Accuracy improvement

---

### Scene 5: Signal Loss Demo (2:35 - 3:20)
**Duration:** 45 seconds

**What to show:**
1. Normal camera operation
2. Kill camera feed (simulate signal loss)
3. Dashboard immediately flags "CAM-01: SIGNAL LOST"
4. Severity: CRITICAL

**Narration:**
> "Now watch what happens when a camera feed goes down. Signal loss is itself an alert. A blinded or jammed camera triggers escalation, not silence. This is a 20-minute build with outsized impact on security credibility."

**Key points to highlight:**
- Normal operation
- Signal loss detection
- Critical severity alert
- "Signal loss is itself an alert"

---

### Scene 6: Hash Chain Demo (3:20 - 3:50)
**Duration:** 30 seconds

**What to show:**
1. Show event log
2. Show hash chain verification
3. Tamper with one event
4. Show chain breaks

**Narration:**
> "Each event record includes a SHA-256 hash of the previous record. If anyone edits a past record, the chain breaks. 10 lines of code. But it answers a question every security judge will ask: how do we know nobody edited the logs?"

**Key points to highlight:**
- Hash chain structure
- Verification process
- Tamper detection
- Security credibility

---

### Scene 7: Closing (3:50 - 4:00)
**Duration:** 10 seconds

**What to show:**
- Return to title screen
- Show "Built for the actual constraint"

**Narration:**
> "IBVAP - designed for the actual constraint, not the demo constraint."

---

## 🎨 Video Editing

### Combine All Scenes

```bash
python scripts/edit_demo.py --action compile \
    --videos 01_title.mp4 02_virtual_fence.mp4 03_anpr_consensus.mp4 \
             04_signal_loss.mp4 05_hash_chain.mp4 \
    --output-file ibvap_demo.mp4
```

### Add Title Card

```bash
python scripts/edit_demo.py --action title \
    --videos ibvap_demo.mp4 \
    --title "IBVAP Demo" \
    --output-file ibvap_final.mp4
```

### Add Text Overlay

```bash
python scripts/edit_demo.py --action overlay \
    --videos ibvap_final.mp4 \
    --text "SIH 2026 - Problem Statement SIH26187" \
    --output-file ibvap_presentation.mp4
```

---

## 📝 Recording Tips

### Before Recording
- [ ] Close all unnecessary applications
- [ ] Turn off notifications (Do Not Disturb mode)
- [ ] Set screen resolution to 1920x1080
- [ ] Clean up desktop background
- [ ] Have all demo files ready
- [ ] Practice the narration 3-5 times

### During Recording
- [ ] Speak clearly and at moderate pace
- [ ] Pause briefly between sections
- [ ] Use mouse to highlight key areas
- [ ] Avoid filler words ("um", "uh", "like")
- [ ] Stay within time limits

### After Recording
- [ ] Review for errors
- [ ] Check audio quality
- [ ] Verify timing
- [ ] Add captions if needed
- [ ] Export in MP4 format (H.264)

---

## 🎤 Narration Script

### Complete Narration (4 minutes)

```
[0:00-0:05] Title Screen
"IBVAP - Intelligent Border Video Analytics Platform. Every AI-CCTV platform 
on the market assumes good bandwidth, good cameras, and infinite trust in 
every alert. Border posts have none of those three."

[0:05-0:35] Architecture
"We designed a three-tier system. Tier 1: Jetson Orin Nano at $250 for full 
AI at the camera site. Tier 2: $20-30 microcontroller for remote cameras. 
Tier 3: Command Center that aggregates everything. Only metadata travels 
upstream, never video."

[0:35-1:35] Virtual Fence
"Watch this. I'm drawing a virtual fence on the camera view. When someone 
crosses this boundary, the system fires an alert in under 3 seconds. Notice 
the explanation field - the operator sees WHY it fired, not just a bounding 
box. This is our answer to the false-positive fatigue problem."

[1:35-2:35] ANPR
"Single-frame OCR on border plates fails constantly. Watch what happens with 
our multi-frame approach. Frame 1: BR12AB3456. Frame 2: BR12AB3458. Frame 3: 
BR12AB3456. The system takes majority vote: BR12AB3456. This is genuinely 
better than single-frame in front of judges."

[2:35-3:20] Signal Loss
"Now watch what happens when a camera feed goes down. Signal loss is itself 
an alert. A blinded or jammed camera triggers escalation, not silence. This 
is a 20-minute build with outsized impact on security credibility."

[3:20-3:50] Hash Chain
"Each event record includes a SHA-256 hash of the previous record. If anyone 
edits a past record, the chain breaks. 10 lines of code. But it answers a 
question every security judge will ask: how do we know nobody edited the logs?"

[3:50-4:00] Closing
"IBVAP - designed for the actual constraint, not the demo constraint."
```

---

## 📁 File Structure

After recording, your directory should look like:

```
SIH2026/
├── demo_videos/
│   ├── 01_title.mp4
│   ├── 02_virtual_fence.mp4
│   ├── 03_anpr_consensus.mp4
│   ├── 04_signal_loss.mp4
│   └── 05_hash_chain.mp4
├── final_videos/
│   ├── ibvap_demo.mp4
│   └── ibvap_presentation.mp4
└── scripts/
    ├── record_demo.py
    ├── screen_recorder.py
    └── edit_demo.py
```

---

## ❓ Troubleshooting

### Video quality is poor
- Ensure screen resolution is 1920x1080
- Check recording FPS (should be 30)
- Use lossless codec if possible

### Audio is out of sync
- Record audio separately
- Sync in post-production
- Use clapperboard or visual cue

### Recording is choppy
- Close other applications
- Reduce recording FPS
- Check system resources

### File is too large
- Use H.264 codec
- Reduce bitrate
- Compress with FFmpeg: `ffmpeg -i input.mp4 -crf 23 output.mp4`

---

*Guide Version: 1.0*
*Last Updated: 2026-08-29*
