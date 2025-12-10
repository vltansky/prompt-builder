# Agent Instructions: Using the refs Folder for Prompt Research

**Purpose:** Guide AI agents to leverage the `temp/` folder when writing, improving, or analyzing prompts

──────────

## Critical Rules

1. **ALWAYS** search the `temp/` folder before writing new prompts
2. **ALWAYS** reference similar examples from `temp/` when improving prompts
3. **NEVER** write prompts from scratch without checking `temp/` first
4. **CITE** specific files from `temp/` when referencing techniques or patterns

──────────

## How to Use the refs Folder

### Step 1: Identify Your Prompt Type

Before writing, determine:
- **Domain:** Technical, creative, educational, security, etc.
- **Provider:** OpenAI, Anthropic, Google, etc.
- **Purpose:** System prompt, jailbreak, domain expert, generator, etc.

### Step 2: Search Across Repositories

The `temp/` folder contains multiple repositories with different focuses:
- **System prompts** - Search across all repos for provider-specific examples
- **Domain-specific prompts** - Look for specialized prompts matching your domain
- **Prompt engineering techniques** - Review advanced patterns and structures
- **Prompt generators** - Find templates and generators for creating prompts
- **Context engineering** - Study structured approaches to prompt design

Use semantic search across the entire `temp/` folder to find relevant examples, then explore specific repositories that match your needs.

### Step 3: Extract Patterns

When reviewing examples, identify:
- **Structure:** How prompts are organized (sections, hierarchy)
- **Tone:** Formal vs casual, directive vs collaborative
- **Techniques:** Role-playing, step-by-step reasoning, examples, constraints
- **Formatting:** Markdown usage, code blocks, lists, emphasis

### Step 4: Apply and Adapt

- **Don't copy verbatim** - Adapt patterns to your specific needs
- **Combine techniques** - Merge successful patterns from multiple examples
- **Cite sources** - Reference specific files when explaining your approach

──────────

## Examples

### Example 1: Writing a System Prompt

**Task:** Create a system prompt for a coding assistant

**Process:**
1. Search `temp/` for "coding assistant" or "code" prompts
2. Review examples from multiple repositories
3. Compare structures and extract common patterns
4. Adapt to your specific requirements

### Example 2: Improving an Existing Prompt

**Task:** Enhance a prompt's clarity and effectiveness

**Process:**
1. Search `temp/` for prompt evaluation and enhancement techniques
2. Review examples of well-structured prompts
3. Apply proven enhancement techniques from multiple sources

### Example 3: Domain-Specific Prompt

**Task:** Create a prompt for a specific domain (e.g., email writing)

**Process:**
1. Search `temp/` for domain-specific examples (e.g., "email writer")
2. Review multiple examples from different repositories
3. Extract domain-specific patterns and adapt

──────────

## Search Strategy

### Use Semantic Search

When searching the `temp/` folder, use queries like:
- "How are system prompts structured?"
- "What techniques do domain expert prompts use?"
- "How do prompt generators work?"
- "What patterns appear in high-quality prompts?"

### Use File Search

For specific domains or providers:
- Search by filename patterns: `*GPT*.md`, `*Prompt*.md`, `*claude*.md`
- Look in specific subdirectories based on your needs

──────────

## Output Format

When referencing the `temp/` folder:

1. **Cite specific files:** "Based on patterns from `temp/[repository]/[path]/[file].md`"

2. **Explain adaptations:** "Adapted the structure from examples in `temp/[repository]/` but simplified for our use case"

3. **Note techniques:** "Used techniques found across multiple repositories in `temp/`"

──────────

## Quality Checklist

Before finalizing a prompt, verify:
- ✓ Searched the `temp/` folder for similar examples
- ✓ Extracted and applied relevant patterns
- ✓ Cited sources when referencing techniques
- ✓ Adapted rather than copied verbatim
- ✓ Combined best practices from multiple examples

──────────

## Remember

- **The `temp/` folder contains multiple repositories with thousands of proven prompts** - Don't reinvent the wheel
- **Patterns > Content** - Focus on structure and techniques, not exact wording
- **Adapt, don't copy** - Use examples as inspiration, not templates
- **Cite your sources** - Help others understand your approach

**The goal:** Leverage the collective wisdom in the `temp/` folder to write better prompts faster.
