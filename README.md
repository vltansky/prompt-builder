# Prompt Builder

A reference library for AI agents to write better prompts by learning from thousands of real-world examples.

## What is this?

This project aggregates the best prompt engineering resources from GitHub into a single searchable collection. When an AI agent needs to write or improve a prompt, it searches this library first to find proven patterns and techniques.

## Quick Start

```bash
# Clone the reference repositories
node scripts/clone-repos.js

# The temp/ folder now contains ~60k+ prompt examples
```

## How It Works

1. **Reference Library** (`temp/`) - Contains cloned repositories with system prompts, jailbreaks, prompt generators, and context engineering guides
2. **Agent Instructions** (`AGENTS.md`) - Rules for AI agents on how to use the library
3. **User Prompts** (`user/`) - Your custom prompts and instructions

## Included Repositories

| Repository | Description | Stars |
|------------|-------------|-------|
| [Awesome_GPT_Super_Prompting](https://github.com/CyberAlbSecOP/Awesome_GPT_Super_Prompting) | Jailbreaks, prompt leaks, security research | 3.4k |
| [chatgpt_system_prompt](https://github.com/LouisShark/chatgpt_system_prompt) | GPT system prompts collection | 9.9k |
| [CL4R1T4S](https://github.com/elder-plinius/CL4R1T4S) | Leaked prompts from ChatGPT, Claude, Cursor, Devin | 12.2k |
| [Context-Engineering](https://github.com/davidkimai/Context-Engineering) | Structured context engineering framework | 7.9k |
| [system_prompts_leaks](https://github.com/asgeirtj/system_prompts_leaks) | Extracted prompts from ChatGPT, Claude, Gemini | 24.2k |
| [TheBigPromptLibrary](https://github.com/0xeb/TheBigPromptLibrary) | Prompts, system prompts, LLM instructions | 4.3k |

## Usage

### For AI Agents

Add `AGENTS.md` to your agent's context. It instructs the agent to:

1. Search `temp/` before writing any prompt
2. Extract patterns from similar examples
3. Cite sources when referencing techniques
4. Adapt rather than copy verbatim

### For Humans

Browse the `temp/` folder to find:

- **System prompts** - How major AI products are prompted
- **Jailbreaks** - Techniques for bypassing restrictions (for security research)
- **Prompt generators** - Templates for creating effective prompts
- **Context engineering** - Advanced techniques for structuring AI context

## Project Structure

```
prompt-builder/
├── AGENTS.md              # Instructions for AI agents
├── scripts/
│   └── clone-repos.js     # Script to fetch reference repos
├── temp/                  # Reference library (gitignored)
│   ├── Awesome_GPT_Super_Prompting/
│   ├── chatgpt_system_prompt/
│   ├── CL4R1T4S/
│   ├── Context-Engineering/
│   ├── system_prompts_leaks/
│   └── TheBigPromptLibrary/
└── user/                  # Your custom prompts
```

## Updating References

Re-run the clone script to get the latest versions:

```bash
node scripts/clone-repos.js
```

This removes existing repos and clones fresh copies (shallow clone, no git history).

## License

This project aggregates content from multiple repositories, each with their own licenses. See individual repositories for their licensing terms.
