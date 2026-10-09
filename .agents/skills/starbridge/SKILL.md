---
name: starbridge
description: Send decisions, questions, and approvals to the user's mobile phone or browser via Starbridge, and track long-running commands. Use when the user is away from their keyboard, needs to make an architectural choice, approve sensitive operations, or monitor long test/build executions.
---

# Starbridge Integration Skill for Antigravity

This skill enables Antigravity to communicate with the user's paired devices (phone, tablet, or web browser) via **Starbridge**. It allows you to ask multiple-choice questions, wait for one-tap answers, and broadcast long-running command progress.

---

## 1. When to Use Starbridge

Use Starbridge when:
1. **The user is away or requests remote notifications** ("Ping my phone", "I'm stepping away", "Notify me when tests finish").
2. **Key architectural or destructive decisions arise** during long autonomous workflows (e.g., choosing between two major implementations, deleting data, force-pushing).
3. **Running lengthy commands** (e.g., test suites, heavy builds, asset imports) that the user wants to monitor remotely.

---

## 2. Asking Questions & Waiting for Decisions

### Synchronous Ask (Blocks until the user taps an option)
To ask a question and immediately wait for the user's answer:

```powershell
starbridge ask --question "Which physics approach should we take for dash collisions?" `
  --option "Raycast Segment" `
  --option "ShapeCast2D Swept" `
  --recommended "Raycast Segment" `
  --context "Raycast Segment avoids tunneling through high-speed projectiles with zero overhead." `
  --wait
```

### Options Breakdown:
* `--question <text>`: The main question (keep concise and actionable).
* `--option <text>`: 2 to 4 selectable choices (omit options for free-text response).
* `--recommended <text>`: Highlights the recommended option.
* `--context <text>`: Extra explanation or trade-offs for each option.
* `--waiting`: Flags the question as actively blocking agent progress (triggers high-priority push).
* `--wait`: Synchronously blocks until the user submits an answer, then prints it to stdout.
* `--timeout <duration>`: Optional timeout (e.g., `10m`, `1h`).

### Asynchronous Ask & Manual Wait
For workflows where you can continue background preparation while waiting:

```powershell
# 1. Post the question and capture the decision ID
$decisionId = (starbridge ask --question "Should we proceed with database migration?" --option "Yes" --option "No").Trim()

# 2. Mark that you are blocked on this decision
starbridge waiting $decisionId

# 3. Wait for the answer when ready
$answer = starbridge wait $decisionId
```

---

## 3. Streaming Long-Running Commands to Phone

Wrap long-running tasks with `starbridge run`. This streams execution time, progress indicators, and pass/fail status directly to the user's mobile device:

```powershell
starbridge run --title "Build / Test Suite" --reason "Running complete verification" -- npm test
```

---

## 4. Environment & Troubleshooting

On Windows PowerShell, ensure Node can verify SSL certificates cleanly:
```powershell
$env:NODE_OPTIONS = "--use-system-ca"
```

To verify pairing status:
```powershell
starbridge config
```
If not paired, run `starbridge pair` or `starbridge setup`.
