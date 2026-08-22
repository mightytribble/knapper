# Knapper MCP Server Setup

## Install

```bash
brew install mightytribble/tap/knapper
knapper index /path/to/documents
```

## Configure MCP Client

**Claude Code** — register with the CLI:

```bash
claude mcp add --scope user knapper -- knapper serve
```

Or add it by hand to `~/.claude.json` (user scope) or a project's `.mcp.json`
(project scope). 

```json
{
  "mcpServers": {
    "knapper": { "type": "stdio", "command": "knapper", "args": ["serve"] }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "knapper": { "command": "knapper", "args": ["serve"] }
  }
}
```

## HTTP Mode

```bash
knapper serve --http              # Port 3000
```
