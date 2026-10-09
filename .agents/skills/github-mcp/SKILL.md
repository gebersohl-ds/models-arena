---
name: github-mcp
description: Automates GitHub repository operations via GitHub MCP, including opening PRs, managing Git tags for model checkpoints, creating issues, and publishing tournament release assets and leaderboard charts.
---

# GitHub MCP Integration Skill (`github-mcp`)

This skill empowers Antigravity to interact natively with your GitHub repositories using the GitHub MCP server (`@modelcontextprotocol/server-github`).

---

## 1. When to Use This Skill

Activate this skill when:
1. **Model Checkpoint Tagging:** Tagging a newly crowned RL model champion (e.g., `v0.2-ppo-matador`) or cutting a game build release.
2. **Automated Leaderboard Publishing:** Committing updated Elo tournament ratings and leaderboard graphics directly to the repo.
3. **Pull Request Workflows:** Opening PRs for completed features, benchmark reports, or bug fixes with structured descriptions and test logs.
4. **Issue Tracking:** Creating, tagging, or closing issues for physics regressions or test suite failures.

---

## 2. Typical Operations & Workflows

### A. Publishing Elo Tournament Results & Tagging Champion
1. Run tournament evaluation locally.
2. Commit leaderboard chart or CSV to branch:
   ```json
   call_mcp_tool(
     ServerName="github",
     ToolName="push_files",
     Arguments={
       "owner": "xSyl0",
       "repo": "my-rl-project",
       "branch": "main",
       "files": [{"path": "benchmarks/elo_leaderboard.md", "content": "# Elo Leaderboard..."}],
       "message": "docs: update Elo leaderboard tournament results"
     }
   )
   ```
3. Tag the release or checkpoint via Git or GitHub releases:
   ```bash
   git tag -a v0.2-ppo-matador -m "Trained PPO Champion v0.2 with 85% winrate"
   git push origin v0.2-ppo-matador
   ```

### B. Opening Automated Pull Requests
Open PRs with formatted test summaries:
```json
call_mcp_tool(
  ServerName="github",
  ToolName="create_pull_request",
  Arguments={
    "owner": "xSyl0",
    "repo": "my-rl-project",
    "title": "feat: optimize spatial grid collision detection",
    "head": "feature/spatial-grid",
    "base": "main",
    "body": "## Summary\n- Replaced O(N^2) collision loop with uniform spatial hash grid.\n- Stepping throughput increased from 1,800 to 58,000 steps/sec."
  }
)
```
