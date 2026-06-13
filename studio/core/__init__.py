"""Core domain vocabulary shared by every step.

These are the fundamental objects the whole system speaks in — the thing being
optimized and the data passed between steps — independent of any single step:

* ``harness``  — the Harness: a directory of editable files (the optimization target).
* ``parts``    — the seven editable part types + the PartMap labeling.
* ``state``    — run/workspace state the orchestrator owns.
* ``observe``  — the progress log emitted as the run proceeds.
* ``evidence`` — structured failure evidence (the benchmark→optimizer bridge).
"""
