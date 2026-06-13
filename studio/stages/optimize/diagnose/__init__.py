"""Diagnose phase: run the harness on held_in, turn failures into addressable
patterns. ``runner`` executes tasks and yields Failures; ``diagnoser`` clusters
them into patterns with a blamed part for the proposer to act on."""
