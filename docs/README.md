# FCA Drift Detection System - Complete Documentation

**Welcome to the comprehensive documentation of the FCA Drift Detection System** 🚀

---

## 📚 Documentation Structure

This folder contains complete documentation for understanding and using the FCA Drift Detection System. Choose your starting point based on your needs:

### 🎯 For Different Audiences

**👨‍💼 Executive/Manager** (5 min read)

- Start: [QUICK_START.md](QUICK_START.md) - "What & Why" section
- Takeaway: Understand the problem and solution

**👨‍💻 Developer (New to project)** (30 min read)

```
1. QUICK_START.md (5 min)          ← The problem & basic flow
2. SYSTEM_OVERVIEW.md (15 min)     ← Architecture & components
3. EXAMPLES.md (10 min)            ← How to use it
```

**🔬 Computer Scientist** (60+ min read)

```
1. QUICK_START.md (5 min)                    ← Problem & solution
2. SYSTEM_OVERVIEW.md (15 min)               ← Architecture
3. COMPLETE_SYSTEM_DOCUMENTATION.md (30 min) ← All algorithms & details
4. EXAMPLES.md (10 min)                      ← Practical code
```

**🐛 Debugger** (20 min read)

```
1. Directly to: COMPLETE_SYSTEM_DOCUMENTATION.md → "Виявлені проблеми і виправлення"
2. Then: Find your specific issue in the section
3. Check: SSOT synchronization (most common issue)
```

---

## 📖 Documents Overview

### 1. **INDEX.md** (This is the navigation hub)

- **Purpose:** Help you find what you need
- **Contains:**
  - Quick navigation links
  - Topic-based search guide
  - FAQ section
  - File structure map
- **Read when:** You need to find something but don't know where

### 2. **QUICK_START.md** (5-10 minutes)

- **Purpose:** Immediately understand the system
- **Contains:**
  - What problem we solve
  - 3 main components (Detector, Aggregator, Reports)
  - 4 drift types
  - Key parameters
  - 3 main bug fixes
- **Read when:** You're new to the project

### 3. **SYSTEM_OVERVIEW.md** (15-20 minutes)

- **Purpose:** Understand architecture & design
- **Contains:**
  - End-to-end pipeline diagram
  - Detailed component descriptions
  - Data structures
  - SSOT principle
  - Parameter recommendations
- **Read when:** You need to understand how pieces fit together

### 4. **COMPLETE_SYSTEM_DOCUMENTATION.md**⭐ (60 minutes)

- **Purpose:** Full technical reference
- **Contains:**
  - Everything from above, but in detail
  - Pseudocode for algorithms
  - Step-by-step traces
  - Type classification rules
  - Detailed problem explanations
  - Code examples
- **Read when:** You need to understand the complete system or debug issues

### 5. **EXAMPLES.md** (10-15 minutes)

- **Purpose:** Learn by doing
- **Contains:**
  - 8 practical code examples
  - From basic usage to advanced customization
  - Parameter tuning guide
  - Performance metrics
  - SSOT verification
- **Read when:** You want to use the code directly

---

## 🔍 Quick Reference

### Problem-Solving Guide

| I need to...                       | Start with...                                                 |
| ---------------------------------- | ------------------------------------------------------------- |
| Understand what this system does   | QUICK_START.md                                                |
| Understand how it works internally | SYSTEM_OVERVIEW.md + COMPLETE_SYSTEM_DOCUMENTATION.md         |
| See code examples                  | EXAMPLES.md                                                   |
| Debug an issue                     | COMPLETE_SYSTEM_DOCUMENTATION.md → Issues section             |
| Find something specific            | INDEX.md                                                      |
| Tune parameters                    | SYSTEM_OVERVIEW.md → Параметрізація + EXAMPLES.md → Example 4 |
| Understand SSOT principle          | SYSTEM_OVERVIEW.md → SSOT section + EXAMPLES.md → Example 7   |

### Key Concepts Map

```
PROBLEM
  └─ Raw drift points (55) vs. meaningful episodes (8)
     └─ Need to aggregate raw detections intelligently

SOLUTION
  ├─ FCADriftDetector
  │   └─ Finds drift points using concept lattice comparison
  │       └─ Produces: drift_indices []
  │
  ├─ DriftAggregatorV2
  │   └─ Merges detections with temporal clustering
  │       └─ Produces: merged_episodes [] (SSOT)
  │
  └─ Graphics & Reports
      └─ Both use merged_episodes for consistent output
          └─ Produces: Synchronized HTML + PNG

RESULT
  └─ Consistent, interpretable drift detection
```

---

## 📊 Algorithm Highlights

### 1. Adaptive Thresholding

```
theta_adaptive = max(theta_baseline, μ + α·σ)
                              ↑↑↑↑↑↑↑↑↑↑↑↑↑
                        Prevents threshold collapse
```

**Impact:** 87.5% fewer false positives

### 2. Temporal-Gap Episodic Clustering

```
IF time_since_last > cooldown AND gap > merge_gap:
  Start new episode
ELSE:
  Continue current episode
```

**Impact:** Groups 55 raw detections → 8 meaningful episodes

### 3. Z-Score Type Classification

```
z = (delta_L - median) / MAD

Type = SUDDEN      if z ≥ 2.5 AND isolated
      | INCREMENTAL if z < 1.5 AND monotonic
  | GRADUAL     if 1.5 ≤ z < 2.5 AND vicinity_count ≥ 3
      | UNKNOWN     otherwise
```

**Impact:** Automatic drift type identification

---

## 🛠 System Status

| Component            | Status        | Notes                       |
| -------------------- | ------------- | --------------------------- |
| FCA Detector         | ✅ Production | With adaptive threshold fix |
| Drift Aggregator     | ✅ Production | SSOT enforced               |
| Type Classification  | ✅ Production | 4-type system               |
| Graphics Export      | ✅ Production | Drift count on all 4 graphs |
| HTML Reports         | ✅ Production | Uses merged episodes        |
| SSOT Synchronization | ✅ Verified   | HTML === Graphics           |

---

## 🔗 Interconnections

### How Documents Relate

```
QUICK_START (overview)
    ↓
SYSTEM_OVERVIEW (architecture)
    ├─→ COMPLETE_SYSTEM_DOCUMENTATION (deep dive)
    │   └─→ EXAMPLES (put it into practice)
    │
    └─→ INDEX.md (find specific topics)
```

### How Components Relate

```
FCADriftDetector
    ↓
    produces: drift_indices [3062, 3147, ...]
    ↓
DriftAggregatorV2 (SSOT source of truth)
    ├─→ HTML Report reads from here
    ├─→ Graphics read from here
    │
    └─→ produces: merged_episodes [Episode1, Episode2, ...]
        ├─ Both HTML and Graphics show same count ✓
        └─ Statistics synchronized ✓
```

---

## 📚 Reading Recommendations

### Fast Track (30 minutes)

1. QUICK_START.md - Understand the basics
2. SYSTEM_OVERVIEW.md - See how it works
3. EXAMPLES.md - See it in action

### Standard Track (90 minutes)

1. QUICK_START.md
2. SYSTEM_OVERVIEW.md
3. COMPLETE_SYSTEM_DOCUMENTATION.md
4. EXAMPLES.md with code examples

### Deep Dive (3+ hours)

1. All documents above, section by section
2. Review source code in `src/fca_drift/`
3. Run examples and experiment with parameters
4. Study scientific references cited

---

## 🎓 Learning Path

### Level 1: Conceptual Understanding

- What is FCA drift detection?
- Why do we need it?
- How does it solve the problem?
- **Time: 10 minutes | Sources: QUICK_START.md**

### Level 2: Architectural Understanding

- What are the main components?
- How do they interact?
- What data structures are used?
- What parameters control behavior?
- **Time: 30 minutes | Sources: SYSTEM_OVERVIEW.md, EXAMPLES.md**

### Level 3: Algorithm Understanding

- How does FCA detector work? (step-by-step)
- How does merging algorithm work? (with trace)
- How does type classification work? (rules)
- What are the scientific bases?
- **Time: 60 minutes | Sources: COMPLETE_SYSTEM_DOCUMENTATION.md**

### Level 4: Implementation Understanding

- How is detector implemented in code?
- How is aggregator implemented in code?
- What are edge cases?
- How to debug issues?
- **Time: 90 minutes | Sources: Code + Examples.md**

### Level 5: Customization & Extension

- How to modify parameters?
- How to change type classification rules?
- How to add custom visualizations?
- How to integrate with other systems?
- **Time: 120+ minutes | Sources: EXAMPLES.md + code experimentation**

---

## ❓ FAQ

**Q: Which document should I start with?**
A: QUICK_START.md - it's designed for first introduction.

**Q: How long to become expert?**
A:

- Basic understanding: 30 min (QUICK_START + SYSTEM_OVERVIEW)
- Comfortable usage: 2-3 hours (all docs + examples)
- Modification & debugging: 5+ hours (code deep-dive)

**Q: Where are the algorithms documented?**
A: Pseudocode in SYSTEM_OVERVIEW.md, step-by-step traces in COMPLETE_SYSTEM_DOCUMENTATION.md

**Q: How do I verify SSOT synchronization?**
A: EXAMPLES.md → Example 7: SSOT Verification

**Q: All documents are in Ukrainian, but I prefer English?**
A: Comments in code are bilingual. Algorithm structure is language-independent.

**Q: What if I found a bug?**
A: Check COMPLETE_SYSTEM_DOCUMENTATION.md → "Виявлені проблеми" section. If it's new, see code comments for debugging guidance.

---

## 📈 Version History

| Version | Date         | Status      | Key Changes                         |
| ------- | ------------ | ----------- | ----------------------------------- |
| 1.0     | Mar 15, 2026 | ✅ Complete | Initial comprehensive documentation |

---

## 🚀 Next Steps

1. **Choose your starting point** based on your background
2. **Read documents in recommended order**
3. **Run examples** to see concepts in action
4. **Modify parameters** and observe impacts
5. **Read source code** for implementation details

---

## 📞 Document Navigation

| If you want...      | Click here...                                                        |
| ------------------- | -------------------------------------------------------------------- |
| Quick introduction  | [QUICK_START.md](QUICK_START.md)                                     |
| System architecture | [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)                             |
| Complete reference  | [COMPLETE_SYSTEM_DOCUMENTATION.md](COMPLETE_SYSTEM_DOCUMENTATION.md) |
| Code examples       | [EXAMPLES.md](EXAMPLES.md)                                           |
| Find something      | [INDEX.md](INDEX.md)                                                 |

---

## 🎯 Summary

This documentation package provides:

- ✅ Quick introduction for newcomers
- ✅ Comprehensive reference for experts
- ✅ Practical examples for developers
- ✅ Scientific foundations for researchers
- ✅ Debugging guides for troubleshooting
- ✅ Navigation aids for finding specific topics

**Start with QUICK_START.md and follow the recommended path for your background.** 📚

---

**Happy learning! 🚀**
