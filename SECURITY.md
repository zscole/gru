# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately.

**Do not open a public issue for security vulnerabilities.**

To report a vulnerability:

1. Email: zcole@linux.com
2. Include a description of the vulnerability
3. Include steps to reproduce if possible
4. Include potential impact assessment

You can expect:
- Acknowledgment within 48 hours
- Status update within 7 days
- Credit in the fix announcement (unless you prefer anonymity)

## Security Considerations

When using Gru, be aware of:

- **Agent Permissions**: Agents execute commands with your user permissions. Use `GRU_WORKDIR` to isolate file operations.
- **Supervised Mode**: Default for a reason. Unsupervised and oneshot modes execute without approval.
- **Secret Storage**: Requires `GRU_MASTER_PASSWORD`. Use a strong password.
- **Admin IDs**: Only users in `GRU_ADMIN_IDS` can interact with the bot. Verify these are correct.
- **MCP Servers**: Review `mcp_servers.json` carefully. MCP servers can execute arbitrary code.
