# Axis Direction Bug Investigation - Document Index

This investigation identified a critical systematic error in the MATLAB-to-Python conversion where sum operation axes were swapped throughout the codebase.

## Quick Start

**Read this first:** `/Users/joeloera/Downloads/Replication Folder for Making America Great Again Claude/replication_package/QUICK_REFERENCE.txt`

## Investigation Documents

### 1. Executive Summary
**File:** `EXECUTIVE_SUMMARY.md`
**Purpose:** High-level overview of the bug, its impact, and solution
**Read if:** You want a complete but concise understanding of the issue

### 2. Quick Reference Card
**File:** `QUICK_REFERENCE.txt`
**Purpose:** One-page cheat sheet with the mapping rules and bug patterns
**Read if:** You need a quick reminder while fixing the code

### 3. Detailed Findings
**File:** `AXIS_MAPPING_FINDINGS.md`
**Purpose:** Complete technical analysis with verification details
**Read if:** You want to understand exactly how the bug was discovered and verified

### 4. Visual Guide
**File:** `AXIS_VISUAL_GUIDE.txt`
**Purpose:** Visual diagrams showing how MATLAB dimensions map to NumPy axes
**Read if:** You're a visual learner or need to explain this to others

### 5. Fix Checklist
**File:** `FIXES_REQUIRED.md`
**Purpose:** Complete list of every code location that needs fixing
**Read if:** You're implementing the fixes

## Test Scripts

### 1. General Axis Mapping Test
**File:** `test_axis_mapping.py`
**Purpose:** Verifies the basic MATLAB → NumPy axis mapping rules
**Run:** `python test_axis_mapping.py`

### 2. Y_i Calculation Test
**File:** `test_axis_detailed.py`
**Purpose:** Proves Bug #1 (GDP calculation) with concrete examples
**Run:** `python test_axis_detailed.py`

### 3. ERR1 Calculation Test
**File:** `test_err1.py`
**Purpose:** Proves Bug #2 (equilibrium condition) with concrete examples
**Run:** `python test_err1.py`

## Key Findings Summary

### The Core Issue
The conversion process incorrectly mapped:
- MATLAB dimension 1 → NumPy axis 1 ❌
- MATLAB dimension 2 → NumPy axis 0 ❌

Correct mapping should be:
- MATLAB dimension 1 → NumPy axis 0 ✓
- MATLAB dimension 2 → NumPy axis 1 ✓

### The Golden Rule
```
MATLAB sum(X, 1) = NumPy sum(X, axis=0)  → column totals (imports)
MATLAB sum(X, 2) = NumPy sum(X, axis=1)  → row totals (exports)
```

### Impact
- 5 files affected
- ~12-15 locations requiring fixes
- All results currently wrong due to incorrect equilibrium conditions

### Files Affected
1. `code_python/analysis/main_baseline.py`
2. `code_python/analysis/main_deficit.py`
3. `code_python/analysis/main_regional.py`
4. `code_python/analysis/main_io.py`
5. `code_python/diagnostic_comparison.py`

## Recommended Reading Order

1. **Start here:** `QUICK_REFERENCE.txt` (2 minutes)
2. **Then read:** `EXECUTIVE_SUMMARY.md` (5 minutes)
3. **For details:** `AXIS_MAPPING_FINDINGS.md` (10 minutes)
4. **For visuals:** `AXIS_VISUAL_GUIDE.txt` (5 minutes)
5. **Before fixing:** `FIXES_REQUIRED.md` (reference document)

## Verification Process

1. Read the documentation
2. Run the test scripts to see the bug in action:
   ```bash
   python test_axis_mapping.py
   python test_axis_detailed.py
   python test_err1.py
   ```
3. Review MATLAB code vs Python code side-by-side
4. Apply fixes per `FIXES_REQUIRED.md`
5. Re-run tests to verify corrections
6. Compare Python output with MATLAB reference output

## Original Context

This investigation was triggered by the observation that:
- MATLAB code uses `sum(X_ji,1)'` and `sum(X_ji,2)`
- Python conversion produces completely different results
- The question was whether the axis mapping was correct

The answer: **No, the axes were systematically swapped.**

## File Locations

All investigation documents and test scripts are located in:
```
/Users/joeloera/Downloads/Replication Folder for Making America Great Again Claude/replication_package/
```

## Contact/Notes

- Investigation completed: 2024-12-04
- All findings verified with concrete test cases
- Test scripts can be run independently to verify the issue
- Documentation is self-contained and can be shared independently

---

**Bottom Line:** The Python code has axes swapped in ~12-15 critical locations. The fix is straightforward: change axis=0 to axis=1 (and vice versa) following the patterns documented in `FIXES_REQUIRED.md`.
