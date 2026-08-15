# Working in this repository

## Markdown: one command per ```bash block

Every fenced ` ```bash ` block in this repo's Markdown holds exactly **one**
command the reader can copy-paste and run. Do not chain unrelated setup steps
into one block with newlines, `&&`, or `;`.

- **Wrong**: `python -m venv venv && source venv/bin/activate` in one block,
  or four separate invocations stacked as four lines in one block.
- **Right**: each command gets its own ` ```bash ` fence, in sequence. A
  trailing inline comment on the command's own line is fine (`pytest  # ~20 s`);
  so is a `# why` comment line directly above the command inside that same
  block — neither of those is a second command.
- A single logical command that spans multiple physical lines via a
  trailing `\` (shell line continuation, or a `srun ... python -c "..."`
  multi-line string) is still **one** command and stays in one block.

Why: multi-command blocks read as "paste this whole thing," but a reader
copy-pasting a five-line block has no way to tell which line failed, and a
comment-only line like `# on an Ares access node` reads as part of the paste
instead of as prose. One command per block makes each step individually
copy-pasteable and each failure attributable to a specific line.

This applies to every `.md` file in the repo (`docs/`, `examples/`,
`profiling/`, top-level `README.md`), not just ones you're actively editing —
if you touch a file with an existing multi-command block, split it while
you're there.
