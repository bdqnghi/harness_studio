"""Edit phase: turn a chosen hypothesis into a candidate harness, and validate it
structurally. ``localizer`` picks the evidence-grounded edit targets; ``strategist``
(the coding agent) makes the edit (and cold-start-generates a harness); ``mapper``
labels which files are editable parts; ``shell`` + ``structural_check`` guard that
the candidate stays within the editable surface and still boots."""
