# Internal Research Policy

## Source selection

Internal policy questions should be researched from the indexed knowledge base first. Public and time-sensitive claims must be checked against current external sources.

## Evidence handling

Every factual claim in a research report must retain its source identifier. Knowledge-base evidence uses a `KB:` citation identifier. Exact local-file evidence uses a `FILE:` citation identifier with the source path and line range when available.

## File access

Large files must not be read into the model context in full. Researchers should retrieve relevant chunks first, expand to adjacent chunks when necessary, and use bounded file-range reads only for exact verification.
