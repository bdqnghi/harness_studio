"""Evaluate phase: score the candidate and decide — these get a benchmark, never a
Backend (the acceptance boundary). ``noise_floor`` measures the run-to-run noise the
gain must beat; ``acceptance`` accepts/rejects on NET pooled gain over
held_in ∪ regression; ``deep_auditor`` re-checks the live harness each segment and
rewinds noise-mirage accepts."""
