# IBVAP — SIH 2026 Slide Deck

## Presentation Structure (5 minutes)

---

## Slide 1: Title + One-Liner (0:00 - 0:30)

### IBVAP
**Intelligent Border Video Analytics Platform**

---

> "Every AI-CCTV platform on the market assumes good bandwidth, good cameras, and infinite trust in every alert. Border posts have none of those three. IBVAP is the first platform designed for the actual constraint — not the demo constraint."

---

**Speaker Notes:**
- Start strong with this one-liner
- Make eye contact with judges
- Pause for 2 seconds after the one-liner for impact
- This immediately signals you understand the real problem

---

## Slide 2: The Problem (0:30 - 1:00)

### What's Wrong with the Obvious Solution?

| What Everyone Pitches | Why It Breaks at a Real BOP |
|---|---|
| Stream video to cloud for AI | Border posts have thin, intermittent links. Full HD streams choke. |
| Detect everything, alert on everything | False-positive fatigue is #1 reason surveillance tech gets switched off |
| Claim FRS works day/night | Physics: no IR hardware = near-zero signal at night. Overclaiming collapses. |
| "Support C2 integration" as checkbox | Nobody defined *how*. Unfalsifiable claims lose technical depth points. |
| Single-frame ANPR | Border plates are angled, blurred, low-res. Single-frame OCR fails constantly. |

---

**Speaker Notes:**
- Point to each row as you explain
- Use concrete examples: "A BOP in Ladakh has 2Mbps satellite link — streaming 4 cameras at 1080p needs 20Mbps"
- This table shows you've thought deeper than other teams
- Judges love seeing "what's wrong with the obvious solution"

---

## Slide 3: Our Solution (1:00 - 1:30)

### IBVAP: Tiered, Bandwidth-Honest, Alert-That-Can-Be-Trusted

```
TIER 1 — High-priority BOP (has power + some connectivity)
  IP Camera → Jetson Orin Nano ($150-250)
      → runs: detection, tracking, ANPR, virtual fence
      → outputs: JSON event metadata only
      → MQTT publish to Command Dashboard

TIER 2 — Remote border-road camera (no compute budget)
  IP Camera → $20-30 microcontroller
      → frame-differencing trigger
      → buffer + forward when linked
      → analytics at nearest Tier-1 node

TIER 3 — Command Center
  Aggregates all events → map dashboard
  → cross-camera correlation → audit log → C2 integration
```

**Key Insight: Only METADATA travels upstream, never video**

---

**Speaker Notes:**
- Show the tier diagram clearly
- Emphasize the cost: "$250 for Tier-1, $50 for Tier-2"
- This is 10x cheaper than proprietary systems
- Say out loud: "A Tier-2 site costs under $50 in added hardware"
- This shows you costed the deployment, not just the demo

---

## Slide 4: Live Demo (1:30 - 3:00) — THE MONEY SHOT

### Demo Flow (90 seconds)

**1. Virtual Fence Intrusion (30 sec)**
- Draw polygon on dashboard
- Walk/drive across it
- Alert fires in <3 sec with full explanation
- "Track T-0042 crossed Zone-3 at 1.4 m/s, bearing NE"

**2. Multi-Frame ANPR (30 sec)**
- Vehicle drives through
- Show single-frame OCR vs our consensus voting
- "3/5 frames agree: ABC-1234" vs "random noise"

**3. Signal Loss (30 sec)**
- Kill a camera feed live
- Dashboard immediately flags "Camera 3: signal lost"
- Severity: HIGH (not blank tile)

---

**Speaker Notes:**
- THIS IS THE MOMENT THAT SEPARATES YOU
- Practice this 5+ times before the hackathon
- Have one person drive the demo, another narrate
- Pre-record backup video for every segment
- If demo crashes, switch to backup immediately
- The signal-loss demo is your "wow" moment — no other team will think of it

---

## Slide 5: Why This Wins (3:00 - 3:30)

### What Makes Us Different

✅ **Signal-loss-as-alert** — Jammed cameras trigger escalation, not silence

✅ **Explainable alerts** — Operators see *why* it fired, not just a bounding box

✅ **Honest night limits** — We don't fake FRS; we state the sensor-physics limit

✅ **Tamper-evident hash chain** — SHA-256 chain proves no one edited the logs

✅ **Store-and-forward** — System never goes silent; degrades gracefully

✅ **Cost analysis** — $250/Tier-1 vs $10,000+ proprietary systems

---

**Speaker Notes:**
- Count these off on your fingers
- Make eye contact with each judge as you list each point
- These are your differentiators — memorize them cold
- If a judge asks "what makes you different?" — this is your answer

---

## Slide 6: Tech Stack + Architecture (3:30 - 4:00)

### Technical Architecture

```
DETECTION:    YOLOv8-nano + ByteTrack (45 FPS on $150 Jetson, 85%+ mAP)
ANPR:         PaddleOCR + multi-frame consensus voting
TRANSPORT:    MQTT (IoT standard, low-bandwidth, defense-proven)
BACKEND:      FastAPI + PostgreSQL + Redis
DASHBOARD:    React + Leaflet maps
SECURITY:     SHA-256 hash chain + RBAC + JWT
```

### Alert Schema (shown on screen)
```json
{
  "event_id": "e7f1...",
  "prev_hash": "a92c...",
  "timestamp": "2026-08-24T21:14:03Z",
  "site_id": "BOP-14",
  "camera_id": "CAM-2",
  "event_type": "fence_intrusion",
  "object_class": "person",
  "track_id": "T-0042",
  "zone": "Zone-3",
  "bearing": "NE",
  "speed_mps": 1.4,
  "confidence": 0.91,
  "baseline_deviation": true,
  "clip_ref": "s3://.../e7f1.mp4",
  "severity": "high",
  "explanation": "Track T-0042 crossed virtual fence Zone-3 at 1.4 m/s, bearing NE."
}
```

---

**Speaker Notes:**
- Show the architecture diagram clearly
- Point out: "We use MQTT — already the standard for defense IoT"
- The alert schema shows technical depth
- Judges love seeing concrete JSON — it proves you thought about integration
- Say: "Any C2 system that can subscribe to MQTT topics can consume our alerts"

---

## Slide 7: Roadmap + Close (4:00 - 4:30)

### What's Built vs What's Planned

**BUILT AND DEMO-READY:**
- ✅ Human + vehicle detection/tracking (YOLOv8-nano)
- ✅ Virtual fence intrusion detection
- ✅ ANPR with multi-frame OCR consensus
- ✅ Alert dashboard with map view
- ✅ Signal-loss alerting
- ✅ Tamper-evident hash chain log

**ROADMAP (honest scope):**
- 🔄 Face detection → identity match (full FRS needs IR hardware)
- 🔄 Behavioral baseline learning (show mechanism, not weeks of training)
- 🔄 Cross-camera correlation (heuristic v1, not full re-ID)
- 🔄 C2 system integration (show JSON payload + MQTT structure)

---

### Next Steps
- Pilot at 2 border posts
- Integrate with existing C2 systems
- Submit for BSF evaluation

---

> **"IBVAP — designed for the actual constraint, not the demo constraint."**

---

**Speaker Notes:**
- Be explicit about what's built vs roadmap
- This honesty builds credibility
- Say: "Face detection at range needs IR hardware — we're honest about this"
- The roadmap shows you've thought beyond the hackathon
- Close with the one-liner again for memorability

---

## Slide 8: Team (4:30 - 4:45)

### The Team

| Name | Role | Expertise |
|------|------|-----------|
| [Name 1] | Tech Lead | System Architecture |
| [Name 2] | ML Lead | Computer Vision, YOLOv8 |
| [Name 3] | Backend Lead | FastAPI, PostgreSQL |
| [Name 4] | Frontend Lead | React, UI/UX |
| [Name 5] | DevOps | Docker, Deployment |
| [Name 6] | Research | Documentation, Slides |

---

**Speaker Notes:**
- Keep this brief — 15 seconds max
- Each team member should be visible
- If a judge asks about a specific area, that person answers
- Shows the team is well-rounded

---

## Slide 9: Thank You + Q&A (4:45 - 5:00)

### Thank You

**IBVAP — Intelligent Border Video Analytics Platform**

- 🌐 GitHub: [repo link]
- 📧 Contact: [email]
- 📄 Documentation: [docs link]

---

### Questions?

---

**Speaker Notes:**
- End with confidence
- Smile
- Be ready for Q&A (see Q&A preparation in SIH_PITCH_GUIDE.md)
- If you don't know an answer: "That's a great question — it's on our roadmap"
- Thank the judges for their time

---

## 🎨 Design Guidelines

### Color Scheme
- **Background:** #0d1117 (dark theme)
- **Primary text:** #ffffff
- **Secondary text:** #8b949e
- **Accent:** #58a6ff (blue)
- **Success:** #4caf50 (green)
- **Warning:** #ff9800 (orange)
- **Danger:** #f44336 (red)

### Typography
- **Headers:** Inter, Bold, 36pt
- **Body:** Inter, Regular, 24pt
- **Code:** JetBrains Mono, 18pt
- **Captions:** Inter, Light, 16pt

### Layout Rules
- One idea per slide
- Maximum 3 bullet points per slide
- Use diagrams over text
- Show real numbers ($250, 45 FPS, <3 sec)
- No paragraphs on slides

### Tools
- **Slides:** Google Slides or Figma
- **Diagrams:** Excalidraw (hand-drawn look) or draw.io
- **Screenshots:** Clean dashboard with real data

---

## 📋 Speaker Notes Summary

### Opening (0:00-0:30)
- Start with the one-liner
- Make eye contact
- Pause for impact

### Problem (0:30-1:00)
- Point to the table
- Use concrete examples
- Show you've thought deeper

### Solution (1:00-1:30)
- Show tier diagram
- Emphasize cost: "$250 vs $10,000"
- Say: "Only metadata travels upstream"

### Demo (1:30-3:00)
- Practice 5+ times
- Have backup video ready
- Signal-loss is your "wow" moment
- If crash → switch to backup immediately

### Why This Wins (3:00-3:30)
- Count off 6 differentiators
- Make eye contact with each judge
- Memorize these cold

### Tech Stack (3:30-4:00)
- Show architecture diagram
- Point out MQTT integration
- Show alert schema JSON

### Roadmap (4:00-4:30)
- Be honest about what's built vs planned
- This builds credibility
- Close with one-liner again

### Team (4:30-4:45)
- Keep brief — 15 seconds
- Each member visible

### Thank You (4:45-5:00)
- End with confidence
- Smile
- Ready for Q&A

---

## 🎯 Key Phrases to Memorize

1. **Opening:** "Every AI-CCTV platform assumes good bandwidth, good cameras, and infinite trust in every alert."

2. **Architecture:** "Only metadata travels upstream, never video."

3. **Cost:** "A Tier-1 site costs under $250; a Tier-2 site costs under $50 — both are 10x cheaper than proprietary systems."

4. **Signal Loss:** "Signal loss is itself an alert. A blinded camera triggers escalation, not silence."

5. **Honesty:** "Without IR hardware, we don't claim face-level ID at night — that's a sensor-physics limit, not a software gap."

6. **Close:** "IBVAP — designed for the actual constraint, not the demo constraint."

---

## 📊 Timing Breakdown

| Section | Start | End | Duration |
|---------|-------|-----|----------|
| Title + One-Liner | 0:00 | 0:30 | 30 sec |
| Problem Statement | 0:30 | 1:00 | 30 sec |
| Our Solution | 1:00 | 1:30 | 30 sec |
| **Live Demo** | **1:30** | **3:00** | **90 sec** |
| Why This Wins | 3:00 | 3:30 | 30 sec |
| Tech Stack | 3:30 | 4:00 | 30 sec |
| Roadmap + Close | 4:00 | 4:30 | 30 sec |
| Team | 4:30 | 4:45 | 15 sec |
| Thank You + Q&A | 4:45 | 5:00 | 15 sec |
| **Total** | **0:00** | **5:00** | **5 min** |

**Buffer:** 30 seconds (built into timing)

---

*Slide Deck Version: 1.0*
*Last Updated: 2026-08-29*
*For: SIH 2026 — Problem Statement SIH26187*
