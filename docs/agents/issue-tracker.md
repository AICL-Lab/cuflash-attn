# Issue Tracker Configuration

## Tracker Type

**GitHub Issues**

## Repo

`AICL-Lab/cuflash-attn`

## How Skills Interact

Skills that create, read, or update issues use the `gh` CLI:

- Create issue: `gh issue create --title "..." --body "..."`
- List issues: `gh issue list --state open`
- View issue: `gh issue view <number>`
- Add label: `gh issue edit <number> --add-label "needs-triage"`
- Close issue: `gh issue close <number>`

## Prerequisites

- `gh` CLI must be installed and authenticated
- User must have write access to the repo
