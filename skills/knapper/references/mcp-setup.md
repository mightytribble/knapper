# Knapper MCP Server Setup

## Install

```bash
brew install mightytribble/tap/knapper
knapper index /path/to/documents
```

## Configure MCP Client

**Claude Code** (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "knapper": { "command": "knapper", "args": ["serve"] }
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
