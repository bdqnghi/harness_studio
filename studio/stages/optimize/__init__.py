"""Step 4 — optimize: the deterministic hypothesis-tree loop.

The **inner loop** (per round, ``orchestrator._round_tree``):
find failures (``runner``) -> diagnose them into patterns (``diagnoser``) ->
route patterns onto the hypothesis tree (``idea_tree``) and select/ideate one
hypothesis (``ideator``) -> localize the evidence-grounded edit targets
(``localizer``) -> implement the edit (``strategist``) -> shell + structural
check (``shell``, ``structural_check``) -> accept-or-reject on NET pooled gain
over held_in ∪ regression (``acceptance``) -> snapshot (``snapshotter``).
Rejections distill an ``insight`` back into the tree so dead ideas are never re-bought.

The **outer loop** (per segment): the ``deep_auditor`` re-checks the live harness
on held_in with fresh rollouts, rewinding noise-mirage accepts. ``noise_floor``
measures the noise floor the acceptance check must beat; ``mapper`` labels editable
parts; ``health`` tracks loop health.

The acceptance boundary is sacred: the acceptance check and deep auditor get a
benchmark, never a Backend — no AI proposal becomes a mutation except through it.
"""
