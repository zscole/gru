# Ralph Wiggum Loops

Ralph Wiggum is an iterative AI development methodology integrated into Gru. It enables autonomous, long-running development cycles where AI agents continuously refine their work until completion.

## Philosophy

Named after the Simpsons character, Ralph embodies persistent iteration despite setbacks. The core principle: **Iteration > Perfection**. Rather than attempting perfect execution in one shot, Ralph loops allow AI agents to progressively improve their work through multiple attempts.

## Quick Start

```bash
# Basic usage
gru ralph "Build a REST API with user authentication"

# With completion detection
gru ralph "Add tests until coverage > 90%" --completion-promise "COVERAGE_MET"

# Custom iterations
gru ralph "Refactor this codebase" --max-iterations 30

# Named loop for tracking
gru ralph "Optimize performance" --name perf-optimizer
```

## How It Works

1. **Initial Task**: Agent receives the task with Ralph loop instructions
2. **Execution**: Agent works on the task autonomously (no approval prompts)
3. **Iteration Check**: When agent completes:
   - If completion promise found → Loop ends successfully
   - If max iterations reached → Loop ends
   - Otherwise → Spawn next iteration with context
4. **Context Preservation**: Each iteration includes summary of previous work
5. **Progressive Refinement**: Agent builds on prior attempts, fixing issues

## Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-iterations` | Maximum number of iterations | 20 |
| `--completion-promise` | String to detect task completion | None |
| `--name` | Custom name for the loop | ralph-[id] |
| `--model` | AI model to use | Default model |
| `--priority` | Task priority (high/normal/low) | normal |

## Best Use Cases

### Ideal For
- **Large refactors**: Framework migrations, dependency updates
- **Test coverage**: Writing comprehensive test suites
- **Batch operations**: Processing multiple files/records
- **Greenfield projects**: Building from scratch with clear requirements
- **Documentation**: Generating complete docs across a codebase
- **Code cleanup**: Linting, formatting, removing dead code

### Not Recommended For
- **Debugging production issues**: Need human judgment
- **One-shot operations**: Simple tasks that don't benefit from iteration
- **Unclear requirements**: Success criteria must be well-defined
- **Time-sensitive tasks**: Iterations take time

## Completion Promises

Completion promises are exact strings the agent must output to signal completion:

```bash
# Agent must output exactly "TESTS_COMPLETE" to finish
gru ralph "Write unit tests" --completion-promise "TESTS_COMPLETE"

# Multiple conditions require multiple Ralph loops
gru ralph "Task 1" --completion-promise "TASK1_DONE"
gru ralph "Task 2" --completion-promise "TASK2_DONE"
```

**Important**: Completion promises use exact string matching. The agent must output the exact string, not a variation.

## Monitoring

```bash
# Check Ralph loop status
gru status <agent-id>

# View iteration progress
gru logs <agent-id>

# Cancel if needed
gru cancel-ralph <agent-id>
```

## Examples

### Example 1: Test Coverage
```bash
gru ralph "Write tests for all components in src/ until every exported function has at least one test" \
  --max-iterations 30 \
  --completion-promise "ALL_FUNCTIONS_TESTED"
```

### Example 2: Framework Migration
```bash
gru ralph "Migrate from Express to Fastify, updating all routes and middleware" \
  --max-iterations 50 \
  --name express-to-fastify
```

### Example 3: Documentation
```bash
gru ralph "Add JSDoc comments to all public functions" \
  --completion-promise "DOCS_COMPLETE" \
  --max-iterations 25
```

## Tips for Success

1. **Clear Completion Criteria**: Define what "done" looks like precisely
2. **Reasonable Iteration Limits**: Set based on task complexity
3. **Test in Isolated Environments**: Use `--workdir` to isolate changes
4. **Monitor Progress**: Check logs periodically to ensure productive iteration
5. **Combine with Templates**: Use saved templates for repeatable Ralph tasks

## Architecture

Ralph loops are implemented through:
- **CLI Commands**: `ralph` and `cancel-ralph` in the CLI interface
- **Orchestrator Integration**: `spawn_ralph_loop()` manages the lifecycle
- **Background Monitoring**: Async task monitors completion and spawns iterations
- **Metadata Tracking**: Loop state stored in `_ralph_loops` dictionary
- **Context Preservation**: Previous work summary included in next iteration

## Safety Features

- **Iteration Limits**: Prevents infinite loops
- **Cancellation**: Can stop loops at any time
- **Unsupervised Only**: Runs without approval prompts for efficiency
- **Status Tracking**: Monitor progress through standard agent commands
- **Automatic Cleanup**: Resources cleaned up on completion/cancellation

## Troubleshooting

**Loop not progressing?**
- Check logs to see if agent is stuck
- Verify task is clear and achievable
- Consider increasing iteration limit

**Completion promise not detected?**
- Ensure exact string match
- Check agent is outputting the promise
- Verify promise is in final message

**Too many iterations?**
- Refine task to be more specific
- Add intermediate completion promise
- Break into smaller Ralph loops

[Back to Commands](commands.md) | [Back to README](../README.md)