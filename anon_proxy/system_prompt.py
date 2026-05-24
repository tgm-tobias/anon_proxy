"""System prompt injected into requests so upstream models handle placeholders correctly.

The proxy replaces detected PII with opaque tokens of the form `<LABEL_N>`
before the request leaves the device. Upstream models, with no context, often
either invent plausible-sounding values to fill in for the tokens or rewrite
them as `[REDACTED]`-style labels — both ruin the round-trip because the
unmasker won't recognize them. This prompt tells the model what the tokens
mean and to echo them verbatim.

The text is intentionally label-format-agnostic (only the `<LABEL_N>` shape is
fixed) so adding new labels via `config.json` doesn't require editing it.
"""

PLACEHOLDER_SYSTEM_PROMPT = (
    "Some of the user's content has been pre-processed by a local privacy "
    "filter: real names, emails, phone numbers, addresses, dates, account "
    "numbers, and similar private values have been replaced with opaque "
    "placeholder tokens of the form <LABEL_N> (for example <PERSON_1>, "
    "<EMAIL_2>, <PHONE_1>). Treat each placeholder as a stable, opaque "
    "reference to a real value you cannot see — two occurrences of the same "
    "token always refer to the same underlying entity. When you need to "
    "refer to one of these entities in your reply, use the token verbatim. "
    "Do not invent real names, emails, phone numbers, addresses, or other "
    "concrete values to fill in for a placeholder, and do not rewrite "
    "tokens as generic labels like [REDACTED] or 'the user'. The proxy "
    "will substitute the original values back into your response before the "
    "user sees it."
)
