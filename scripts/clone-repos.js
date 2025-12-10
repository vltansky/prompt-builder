import { execSync } from "node:child_process";
import { rmSync, existsSync } from "node:fs";
import { join } from "node:path";

const TEMP_DIR = join(import.meta.dirname, "..", "temp");

const repos = [
  {
    name: "Awesome_GPT_Super_Prompting",
    url: "https://github.com/CyberAlbSecOP/Awesome_GPT_Super_Prompting",
  },
  {
    name: "chatgpt_system_prompt",
    url: "https://github.com/LouisShark/chatgpt_system_prompt",
  },
  {
    name: "CL4R1T4S",
    url: "https://github.com/elder-plinius/CL4R1T4S",
  },
  {
    name: "Context-Engineering",
    url: "https://github.com/davidkimai/Context-Engineering",
  },
  {
    name: "system_prompts_leaks",
    url: "https://github.com/asgeirtj/system_prompts_leaks",
  },
  {
    name: "TheBigPromptLibrary",
    url: "https://github.com/0xeb/TheBigPromptLibrary",
  },
];

for (const repo of repos) {
  const targetPath = join(TEMP_DIR, repo.name);

  if (existsSync(targetPath)) {
    console.log(`Removing existing ${repo.name}...`);
    rmSync(targetPath, { recursive: true, force: true });
  }

  console.log(`Cloning ${repo.name}...`);
  execSync(`git clone --depth 1 ${repo.url} ${targetPath}`, {
    stdio: "inherit",
  });

  const gitDir = join(targetPath, ".git");
  if (existsSync(gitDir)) {
    console.log(`Removing .git from ${repo.name}...`);
    rmSync(gitDir, { recursive: true, force: true });
  }

  console.log(`Done: ${repo.name}\n`);
}

console.log("All repos cloned successfully!");
