# Changelog fragments

Every pull request adds **one file** here named after its PR number, e.g.
`131.md`, containing its changelog line(s):

```
- **ADD:** Short description of the change. ({pull}`131`) by [@user](https://github.com/user)
```

Use the prefix of the PR title (`ADD`, `ENH`, `FIX`, `MNT`, `DOC`, `DEP`, `REL`).
Because each PR writes its own file, changelog entries never conflict between
pull requests. The documentation build collects the fragments into the
"Unreleased" section of `history.md`; at a release,
`python ci/release_changelog.py X.Y.Z` moves them into a new version section
and deletes the fragments.
