# RSS Security Agent — CBAC policy

The agent researches security developments: vulnerabilities, advisories, authentication, authorization and AI or MCP security, using only the tools of the RSS research MCP
server over the feeds this deployment configures. It may list the configured
feeds, fetch the latest articles of a configured feed, search the cached
articles, and read one cached article.

The `feed_id` argument names the feed a call reads: it is the resource, and
only the feeds listed in the trusted data may be granted. The `query`
argument is search text, `limit` is a page size, and `article_id` is an
identifier the server returned in an earlier result; none of them names what
the call acts on, so they are free text.

The agent must never read a feed that is not in the trusted data.
