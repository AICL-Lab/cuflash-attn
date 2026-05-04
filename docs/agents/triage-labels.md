# Triage Label Vocabulary

## Label Mapping

| Role | Label String |
|------|--------------|
| Needs evaluation | `needs-triage` |
| Waiting on reporter | `needs-info` |
| AFK-ready | `ready-for-agent` |
| Needs human | `ready-for-human` |
| Won't fix | `wontfix` |

## State Machine

The `triage` skill moves issues through these states:

1. **New issue** → `needs-triage` (maintainer evaluates)
2. **Needs clarification** → `needs-info` (waiting on reporter)
3. **Fully specified** → `ready-for-agent` or `ready-for-human`
4. **Rejected** → `wontfix`

## Setup

Ensure these labels exist in your GitHub repo. If they don't, the `triage` skill will attempt to create them, or you can create them manually in GitHub Settings → Labels.
