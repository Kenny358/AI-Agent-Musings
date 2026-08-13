# Can an Agent Really Learn? From a Markdown File to a LoRA Adapter

[中文版](README.md)

I keep returning to a simple question. When we say that an agent gets to know its user over time, what exactly has learned?

Suppose a user repeatedly asks the agent to show a diff before editing a file. The agent can save that preference in `USER_PREFERENCE.md` and load it before the next task. Its behavior changes, but the model parameters do not.

Does that count as learning?

At the system level, yes. At the model-training level, the model has merely received another piece of reading material. I use two deliberately informal names to keep these cases apart. One is the open-book approach. The other is the closed-book approach.

Picture an open-book exam. Forgetting one formula is fine if the right reference sheet comes into the room. A math exam gets a formula sheet. A project task gets `PROJECT_RULES.md`. User preferences wait inside `USER_PREFERENCE.md`. The notes still have to match the exam. Bringing a world-history textbook into a calculus test gives you plenty to read and very little help.

That is where the name open-book comes from. Before every task, the agent prepares a cheat sheet for that particular question. Markdown, memory, RAG, and skills can all contribute. The model parameters stay put. Swap the sheet, and the same model can immediately follow a different set of instructions.

A closed-book exam asks more of the student. The notes stay outside, so preparation involves memorizing, practicing, and revisiting mistakes. Human learning involves complex biological changes, including changes in neural connections. The analogy only borrows that intuition. A human brain and a neural network are not the same machine.

Large models do not attend evening study hall. Their preparation is training. Data arrives in batches, gradients update the model step by step, and the parameters change. When the model sits the next exam, some trace of that practice is already in its parameters. The agent no longer needs to hide the old answer sheet under the exam paper.

A closed-book exam still comes with an exam paper, and the model still receives the current prompt. The agent simply stops slipping past experience into the context as a cheat sheet.

```mermaid
flowchart TB
    A["Agent learning"] --> B["Open-book<br/>Context Learning"]
    A --> C["Closed-book<br/>Parameter Learning"]
    B --> B1["MD / Memory"]
    B --> B2["RAG"]
    B --> B3["Rules / Skills"]
    B --> B4["Tool / DB"]
    B1 --> B5["Parameters θ unchanged<br/>Knowledge, rules, preferences, and changing information"]
    B2 --> B5
    B3 --> B5
    B4 --> B5
    C --> C1["LoRA / Adapter<br/>W + ΔW"]
    C --> C2["Full Fine-tuning<br/>W → W′"]
    C1 --> C3["Stable capabilities and behavior"]
    C2 --> C3
```

## The open-book Markdown approach

The open-book side is mostly a job of finding the right notes. The model does not need to memorize user preferences or project rules. The agent retrieves them and places them in the context before the exam begins. Fetch the wrong material, and the model may read it very carefully before answering the wrong question. That material may come from Markdown files, a database, RAG, a knowledge graph, or a skill. Markdown is simply the easiest version to inspect.

```text
User task
  +
USER_PREFERENCE.md
  +
PROJECT_RULES.md
  +
Relevant memories or retrieved documents
  ↓
LLM
```

Expanded into its working parts, the path looks like this.

```mermaid
flowchart TB
    A["Current user prompt"] --> G["Assemble the context"]
    B["Memory / user preferences"] --> G
    C["Rules / Skills"] --> G
    D["Project Markdown"] --> F["Task-specific retrieval"]
    E["DB / RAG / external knowledge"] --> F
    F --> G
    G --> H["LLM inference"]
    H --> I["Agent decisions and tool calls"]
    I --> J["Task outcome"]
    J -. "Verified new experience" .-> B
```

If the model parameters are denoted by $\theta$, then

$$
\theta_{t+1}=\theta_t
$$

The model has not changed. It walks into each exam carrying the right notes.

This approach works well for user preferences, project conventions, explicit policies, and changing knowledge. If a company changes its report format, we edit a file. Retraining a model for that would be wasteful.

A growing context window is not, by itself, a reason to fine-tune. We can retrieve only the relevant passages, compress old memories, and move deterministic steps into tools or workflows. Many apparent capability failures are retrieval or orchestration failures in disguise.

## The closed-book parameter approach

I will call the original base model, before this round of training, the parent model. The closed-book side requires practice. We collect the questions the model handled well, the mistakes it keeps repeating, and the corrections worth keeping. Training then moves the useful part into parameters. LoRA is one way to do this. Roughly speaking, it gives the parent model a replaceable set of answering habits.

An agent can retain task trajectories containing prompts, tool calls, final answers, user edits, and verified outcomes. After careful filtering, those records can support supervised fine-tuning, or SFT, preference optimization, and other post-training methods. LoRA is one practical option.

```mermaid
flowchart TB
    A["A batch of past agent tasks"] --> B["Prompts"]
    A --> C["Tool calls and results"]
    A --> D["Answers"]
    A --> E["User acceptance, edits, or retries"]
    A --> F["Task success or failure"]
    B --> G["Verify, deduplicate, review privacy, and filter"]
    C --> G
    D --> G
    E --> G
    F --> G
    G --> H["Build SFT / preference-training data"]
    H --> I["LoRA Training"]
    I --> J["Parameter update ΔW"]
    K["Frozen parent weights W"] --> L["Agent model version W + ΔW"]
    J --> L
```

For a model weight matrix $W$, LoRA freezes the original weights and learns two smaller low-rank matrices, $A$ and $B$. A common expression is

$$
W'=W+\frac{\alpha}{r}BA
$$

The base weights remain intact while the adapter supplies a learned update. The model may now behave differently even when the old lesson is absent from the prompt. In that sense, the lesson has moved into parameters.

Versioned adapters can be stored beside the parent model.

```text
Qwen Base
  ├── No adapter
  ├── agent-v1
  ├── agent-v2
  └── agent-v3
```

Keeping the base checkpoint and older adapters makes switching and rollback straightforward. It also makes LoRA attractive for low-cost, isolated experiments.

Rollback does not make an adapter inherently safe. It can still overfit, reinforce bad examples, or degrade an older capability. Full fine-tuning can also be rolled back when the original checkpoint has been preserved. The dangerous setup is the one with no versioned checkpoint, no evaluation, and no recovery path.

Closed-book learning can also update the parent weights directly through full fine-tuning. LoRA learns an update stored beside the parent model, while full fine-tuning produces a new set of parent weights. Both are forms of parameter learning. They differ in update scope, training cost, and version isolation.

## Logs are raw material, not a training set

A user accepting an answer is a weak signal. They may simply be tired of revising it. A retry does not prove that the previous answer was entirely wrong.

Agent logs therefore need outcome verification, deduplication, privacy review, and quality filtering before they become training data. High-stakes domains such as medicine and finance also require qualified human review. Letting a model grade its own output and immediately train on that grade is an efficient way to preserve mistakes.

```mermaid
flowchart TB
    A["User task"] --> B["Parent model + current adapter"]
    B --> C["Agent executes the task"]
    C --> D["Immediate experience"]
    C --> E["Task logs and feedback"]
    D --> F["Update MD / Memory<br/>Open-book learning"]
    E --> G["Verify outcomes, deduplicate, review privacy, and filter"]
    G --> H["Training Pool"]
    H --> I{"Enough data<br/>and a stable capability gap"}
    I -- "No" --> H
    I -- "Yes" --> J["Train candidate Adapter v2"]
    J --> K["Benchmark vs v1"]
    K -- "Fail" --> L["Reject candidate"]
    K -- "Pass" --> M["Canary release"]
    M --> N["Production"]
    N --> C
```

Training should not start merely because the system has collected ten thousand prompts. It needs enough high-quality data and evidence of a stable, repeated capability gap. Without both conditions, post-training mostly teaches the model about noise.

## After choosing how to learn, choose when to update

So far, we have answered one question. How does the agent learn? The open-book system changes material outside the parameters. The closed-book system trains parameters.

Engineering introduces another question. When does the update happen? The system may process each new experience as it arrives, or collect a batch and process it later. The first is usually described as real-time or online updating. The second is periodic or batch updating.

```mermaid
flowchart LR
    A["Agent learning"] --> B["How it learns"]
    A --> C["When it updates"]
    B --> B1["Open-book<br/>Context Learning"]
    B --> B2["Closed-book<br/>Parameter Learning"]
    C --> C1["Real-time / online"]
    C --> C2["Periodic / batch"]
    B1 -. "Can combine" .-> C1
    B1 -. "Can combine" .-> C2
    B2 -. "Can combine" .-> C1
    B2 -. "Can combine" .-> C2
```

Timing and method are separate dimensions. Memory can be updated immediately or consolidated periodically. Parameter updates can also occur continuously or in batches.

Updating weights after every agent task is theoretically possible, but difficult to control. Bad feedback, shifting data, and catastrophic forgetting can accumulate quickly.

An asynchronous pipeline is easier to reason about. The production model keeps serving users while a candidate adapter trains in the background. The candidate faces the same benchmark as the current version, followed by a small canary release. Failed candidates are discarded, and production can return to an earlier adapter.

## Keep knowledge open-book and train stable capabilities carefully

The decision cannot be reduced to everyday work using Markdown while medicine and finance use LoRA.

A transaction threshold that requires a second confirmation is an explicit rule. It belongs in a policy engine or external configuration. Regulations, drug information, and market prices change. They belong in retrieval systems, databases, and tools rather than solely in model parameters.

LoRA is a better candidate for stable behavioral patterns that require many examples to describe. Imagine a model that repeatedly answers questions about local files from memory even though it should search them. The prompt is clear, the workflow has been checked, and the same failure remains visible in a fixed evaluation set. That is a capability gap worth considering for post-training.

Medical and financial systems demand stricter validation, auditability, and human oversight. Higher stakes make casual adapter updates less acceptable. Trained capability, current knowledge, deterministic policy, and calculation tools need to work together.

## My decision rule

I start with the simplest test. Can one clear rule solve the problem? If it can, the rule goes into Markdown, a skill, or configuration. When the material grows, retrieval limits what enters the prompt. Deterministic work moves into tools and workflows.

```mermaid
flowchart TB
    A["Use the open-book system by default"] --> B["A repeated error appears"]
    B --> C{"Can one explicit rule solve it"}
    C -- "Yes" --> D["Update MD / Skill / configuration"]
    C -- "No" --> E{"Can deterministic code solve it"}
    E -- "Yes" --> F["Improve the Tool / Workflow"]
    E -- "No" --> G{"Does correct behavior require<br/>many examples to describe"}
    G -- "No" --> H["Keep improving context and retrieval"]
    G -- "Yes" --> I{"Does the error persist<br/>in a fixed evaluation"}
    I -- "No" --> H
    I -- "Yes" --> J["Enter the LoRA Training Pool"]
```

Only a stable failure that survives those changes becomes a LoRA candidate. It also needs enough reviewed examples to describe the desired behavior. A candidate adapter reaches production only after it beats the current version on fixed capability and safety evaluations.

The open-book system learns quickly because an edited document can affect the next task. The closed-book system learns slowly because it requires evidence, training, evaluation, and deployment. One follows change. The other develops stable behavioral habits.
