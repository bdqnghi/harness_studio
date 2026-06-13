"""Record phase: durable bookkeeping as the loop runs. ``snapshotter`` saves a
harness snapshot per round; ``health`` tracks loop health (e.g. consecutive
rejection streaks) so a stuck run can be flagged."""
