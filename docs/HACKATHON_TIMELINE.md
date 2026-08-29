# IBVAP — SIH 2026 Hackathon Timeline & Task Breakdown

## 📅 Pre-Hackathon Preparation (1-2 Weeks Before)

### Week 2: Research & Planning
| Day | Task | Owner | Deliverable |
|-----|------|-------|-------------|
| Day 1 | Finalize problem statement understanding | All | Problem analysis doc |
| Day 2 | Research existing solutions & competitors | Research | Competitive analysis |
| Day 3 | Design system architecture | Tech Lead | Architecture diagram |
| Day 4 | Set up development environment | DevOps | Working dev setup |
| Day 5 | Create project repo & CI/CD | DevOps | GitHub repo |
| Day 6 | Design UI/UX mockups | Designer | Figma/Sketch mockups |
| Day 7 | Team meeting & task assignment | All | Task board |

### Week 1: Setup & Prototyping
| Day | Task | Owner | Deliverable |
|-----|------|-------|-------------|
| Day 1 | Implement YOLOv8 detection | ML Lead | Working detection |
| Day 2 | Implement ByteTrack tracking | ML Lead | Working tracker |
| Day 3 | Create FastAPI backend skeleton | Backend | API endpoints |
| Day 4 | Create React dashboard skeleton | Frontend | Dashboard layout |
| Day 5 | Implement virtual fence logic | ML Lead | Fence detection |
| Day 6 | Implement ANPR engine | ML Lead | OCR consensus |
| Day 7 | Integration test & demo prep | All | Working prototype |

---

## 🎯 Hackathon Day Structure

### Day 1: Core Development (8 hours)

#### Hour 0-2: Setup & Foundation
| Task | Owner | Status |
|------|-------|--------|
| Set up development environment | DevOps | ☐ |
| Clone repos & install dependencies | All | ☐ |
| Finalize architecture decisions | Tech Lead | ☐ |
| Assign specific tasks | All | ☐ |

#### Hour 2-5: Core ML Pipeline
| Task | Owner | Status |
|------|-------|--------|
| Integrate YOLOv8 with ByteTrack | ML Lead | ☐ |
| Implement virtual fence detection | ML Lead | ☐ |
| Implement ANPR with consensus voting | ML Lead | ☐ |
| Test detection accuracy | ML Lead | ☐ |
| Optimize for edge deployment | ML Lead | ☐ |

#### Hour 5-7: Backend & API
| Task | Owner | Status |
|------|-------|--------|
| Complete FastAPI endpoints | Backend | ☐ |
| Implement event ingestion | Backend | ☐ |
| Implement hash chain verification | Backend | ☐ |
| Set up WebSocket for real-time | Backend | ☐ |
| Test API endpoints | Backend | ☐ |

#### Hour 7-8: Frontend Foundation
| Task | Owner | Status |
|------|-------|--------|
| Create dashboard layout | Frontend | ☐ |
| Implement map component | Frontend | ☐ |
| Create alert card component | Frontend | ☐ |
| Connect to backend API | Frontend | ☐ |
| Test real-time updates | Frontend | ☐ |

---

### Day 2: Integration & Features (8 hours)

#### Hour 0-3: Signal Loss & Security
| Task | Owner | Status |
|------|-------|--------|
| Implement signal loss detection | ML Lead | ☐ |
| Create signal loss alerts | Backend | ☐ |
| Add hash chain to all events | Backend | ☐ |
| Test tamper-evident log | Backend | ☐ |
| Implement RBAC | Backend | ☐ |

#### Hour 3-5: Dashboard Features
| Task | Owner | Status |
|------|-------|--------|
| Implement alert filtering | Frontend | ☐ |
| Add site status indicators | Frontend | ☐ |
| Create event detail modal | Frontend | ☐ |
| Add severity color coding | Frontend | ☐ |
| Implement real-time updates | Frontend | ☐ |

#### Hour 5-7: Integration Testing
| Task | Owner | Status |
|------|-------|--------|
| End-to-end testing | All | ☐ |
| Fix critical bugs | All | ☐ |
| Performance optimization | All | ☐ |
| Demo video recording | All | ☐ |
| Backup plan preparation | All | ☐ |

#### Hour 7-8: Polish & Documentation
| Task | Owner | Status |
|------|-------|--------|
| UI polish & styling | Frontend | ☐ |
| API documentation | Backend | ☐ |
| README & setup guide | All | ☐ |
| Final testing | All | ☐ |
| Demo rehearsal | All | ☐ |

---

### Day 3: Demo & Presentation (8 hours)

#### Hour 0-2: Final Preparations
| Task | Owner | Status |
|------|-------|--------|
| Final bug fixes | All | ☐ |
| Demo environment setup | DevOps | ☐ |
| Backup video ready | All | ☐ |
| Presentation slides finalized | All | ☐ |
| Team roles confirmed | All | ☐ |

#### Hour 2-4: Demo Rehearsal
| Task | Owner | Status |
|------|-------|--------|
| Full demo run-through #1 | All | ☐ |
| Timing check (5 min limit) | All | ☐ |
| Q&A preparation | All | ☐ |
| Full demo run-through #2 | All | ☐ |
| Final adjustments | All | ☐ |

#### Hour 4-5: Presentation
| Task | Owner | Status |
|------|-------|--------|
| Setup display/projector | DevOps | ☐ |
| Test audio/video | All | ☐ |
| Final team huddle | All | ☐ |
| **PRESENTATION** | All | ☐ |
| Q&A session | All | ☐ |

#### Hour 5-8: Results & Next Steps
| Task | Owner | Status |
|------|-------|--------|
| Collect feedback | All | ☐ |
| Document lessons learned | All | ☐ |
| Plan next development phase | All | ☐ |
| Celebrate! 🎉 | All | ☐ |

---

## 👥 Team Roles & Responsibilities

### Role Assignments

| Role | Responsibilities | Skills Needed |
|------|------------------|---------------|
| **Tech Lead** | Architecture decisions, code reviews, integration | System design, leadership |
| **ML Lead** | YOLOv8, ByteTrack, ANPR, signal loss detection | Computer vision, Python |
| **Backend Lead** | FastAPI, PostgreSQL, WebSocket, MQTT | Python, databases |
| **Frontend Lead** | React, dashboard, UI/UX | React, CSS, Leaflet |
| **DevOps** | Docker, CI/CD, deployment, demo setup | Docker, Linux |
| **Research** | Competitor analysis, documentation, slides | Research, presentation |

### Communication Plan
- **Daily standup:** 15 min at start of each day
- **Slack channel:** #ibvap-sih2026
- **Code reviews:** All PRs require 1 approval
- **Demo prep:** Daily 30 min practice sessions

---

## 🎯 Success Criteria

### Must-Have (Demo Requirements)
- [ ] YOLOv8 detection working on video feed
- [ ] ByteTrack tracking with persistent IDs
- [ ] Virtual fence intrusion detection
- [ ] ANPR with multi-frame consensus
- [ ] Signal loss alerting
- [ ] Dashboard with map view
- [ ] Alert feed with explanations
- [ ] Hash chain verification demo

### Should-Have (Polish)
- [ ] Real-time WebSocket updates
- [ ] Site status indicators
- [ ] Event detail modal
- [ ] Severity color coding
- [ ] Responsive design

### Nice-to-Have (Extra Credit)
- [ ] MQTT integration demo
- [ ] Role-based access control
- [ ] Configurable retention policies
- [ ] Performance metrics dashboard

---

## 🚨 Risk Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| YOLOv8 fails to load | Low | High | Pre-download model, have backup |
| WebSocket connection drops | Medium | Medium | Auto-reconnect, polling fallback |
| Demo video won't play | Medium | High | Multiple backup videos |
| API server crashes | Low | High | Docker restart, demo mode |
| Laptop crashes | Low | Critical | Backup laptop ready |

### Presentation Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Demo crashes mid-presentation | Medium | Critical | Switch to pre-recorded video |
| Judges ask tough question | High | Medium | Prepared Q&A responses |
| Time runs out | Medium | High | Practice with timer |
| Team member forgets lines | Medium | Medium | Everyone knows the pitch |
| WiFi fails | Low | Medium | Offline demo mode |

---

## 📋 Daily Checklist

### Day 1 Morning
- [ ] Team huddle (15 min)
- [ ] Review day's tasks
- [ ] Set up development environment
- [ ] Start core ML pipeline

### Day 1 Evening
- [ ] Demo working detection
- [ ] Push code to GitHub
- [ ] Update task board
- [ ] Plan Day 2

### Day 2 Morning
- [ ] Team huddle (15 min)
- [ ] Review Day 1 progress
- [ ] Start integration
- [ ] Begin dashboard development

### Day 2 Evening
- [ ] End-to-end test passing
- [ ] Demo video recorded
- [ ] Presentation slides ready
- [ ] Plan Day 3

### Day 3 Morning
- [ ] Team huddle (15 min)
- [ ] Final bug fixes
- [ ] Demo rehearsal #1
- [ ] Q&A preparation

### Day 3 Afternoon
- [ ] Demo rehearsal #2
- [ ] Setup presentation area
- [ ] Final team huddle
- [ ] **PRESENT & WIN! 🏆**

---

## 🎯 Key Metrics to Track

### Development Metrics
- **Commits per day:** Target 10-15
- **PRs merged:** Target 5-8 per day
- **Bugs fixed:** Track daily
- **Test coverage:** Target 70%+

### Demo Metrics
- **Detection accuracy:** Target >85% mAP
- **Alert latency:** Target <3 seconds
- **ANPR accuracy:** Target >90% with consensus
- **Dashboard load time:** Target <2 seconds

### Presentation Metrics
- **Pitch timing:** Target 4:30 (30 sec buffer)
- **Demo timing:** Target 3:00
- **Q&A responses:** Target <10 seconds per answer
- **Team confidence:** Target HIGH

---

## 🏆 Winning Checklist

### Before Hackathon
- [ ] Problem statement fully understood
- [ ] Architecture designed & reviewed
- [ ] Development environment ready
- [ ] Team roles assigned
- [ ] Communication channels set up

### During Hackathon
- [ ] Daily standups on time
- [ ] Code reviews completed
- [ ] Demo tested daily
- [ ] Documentation updated
- [ ] Backup plans ready

### Before Presentation
- [ ] Demo works flawlessly
- [ ] Slides finalized
- [ ] Team knows the pitch
- [ ] Q&A responses prepared
- [ ] Backup video ready

### During Presentation
- [ ] Start with one-liner hook
- [ ] Show architecture diagram
- [ ] Live demo (virtual fence → ANPR → signal loss)
- [ ] Close with roadmap
- [ ] Handle Q&A confidently

### After Presentation
- [ ] Collect feedback
- [ ] Document lessons learned
- [ ] Plan next phase
- [ ] **CELEBRATE! 🎉**

---

*Timeline Version: 1.0*
*Last Updated: 2026-08-29*
*For: SIH 2026 — Problem Statement SIH26187*
