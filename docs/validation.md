# Signal validation

ABFE accepts finite, real numeric, one-dimensional pre-windowed signals.

Validation deliberately rejects:

- row, column, and higher-dimensional arrays;
- empty input;
- NaN and infinity;
- boolean arrays;
- string arrays, including numeric-looking strings;
- complex arrays;
- object and mixed-type arrays.

Parsing text or spreadsheet values belongs in the caller's ingestion pipeline.
Silently converting a boolean mask or string column into a physiological signal
could create plausible but invalid feature values.

`validate_signal` always returns an owned, read-only `float64` array. It never
aliases caller memory. This favors a simple safety contract; it should only be
revisited if future profiling demonstrates a material extraction bottleneck.
