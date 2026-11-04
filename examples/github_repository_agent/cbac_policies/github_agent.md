# GitHub Repository Agent — CBAC policy

The agent analyses one GitHub repository through the read-only tools of the
GitHub MCP server. It may read the repository named in the trusted data, and
only what the user's request asks for: its metadata, directory listings, file
contents, recent commits, open issues and open pull requests.

The `repository` argument names what every call acts on: it is the resource,
and only the configured repository may be granted. The `path` argument of
`directory_listing` and `file_contents` and every `limit` argument are settings
of the call inside that repository; they name nothing on their own.

The agent must never read a repository other than the configured one.
The agent must never read files under `.github/workflows/` or any file whose
name contains `secret`, `token` or `credential`.
