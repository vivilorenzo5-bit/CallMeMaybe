*This project has been created as part of the 42 curriculum by vlourenc.*

# Call Me Maybe - Constrained Decoding for Function Calling

## Description
**Call Me Maybe** is an autoregressive constrained decoding engine built in Python that translates unstructured natural language requests into deterministic, 100% schema-compliant JSON function calls. 

Small Language Models (SLMs), such as `Qwen/Qwen3-0.6B`, struggle significantly with generating structured syntax consistently when relying solely on system prompts, frequently dropping delimiters, hallucinating non-existent keys, or producing malformed JSON. This project bridges that gap by intercepting the neural network's logit distribution at each generation step. By combining a strict Finite State Machine (FSM) grammar and a high-performance Trie prefix tree, the engine eliminates invalid tokens before sampling, mathematically guaranteeing valid JSON output and exact argument schema adherence without external heavyweight inference frameworks.

---

## Instructions

### Prerequisites
- Python >= 3.10
- `uv` package and project manager


## Makefile Automation

The project includes a standard Makefile supporting development workflows:

    make install: Synchronizes environment dependencies using uv sync.

    make run: Executes the default batch pipeline against the provided dataset.

    make debug: Runs the pipeline with Python's interactive debugger (pdb).

    make clean: Removes Python bytecode caches, temporary directories, and coverage artifacts.

    make lint: Validates code style and type safety with flake8 and mypy.

    make lint-strict: Enforces strict flake8 and mypy --strict compliance.

## Algorithm Explanation
The constrained decoding architecture intervenes directly inside the token-by-token autoregressive loop:

	1. Context Initialization: The user prompt and registered function specifications are formatted into the Qwen ChatML template (<|im_start|>system ... <|im_end|>\n<|im_start|>user ...).
	2. Logit Extraction: The raw input sequence is passed to Small_LLM_Model.get_logits_from_input_ids(), returning a 1D vector of 151,643 logits representing the model's unnormalized token likelihoods.
	3. Grammar Enforcement via FSM: The current generation state in JsonFSM queries its structural grammar (e.g., function name, argument key, delimiter, or typed value) to construct an allowlist of valid token IDs.
	4. Logits Masking: LogitsMasker applies vectorized NumPy indexing across the vocabulary. Any token ID not explicitly permitted by the FSM is overwritten with $-\infty$.
	5. Greedy Selection: An argmax operation selects the highest-scoring permitted token. Because $-\infty$ evaluates strictly below any real logit value, forbidden tokens cannot be sampled.
	6. State Transition: The chosen token is consumed by the FSM, updating internal string buffers, advancing grammatical states, and appending the token ID to the running context until FSMStateType.DONE is reached.

## Design Decisions
Trie Prefix Indexing: Matching dynamic substrings against a 151k vocabulary naively using string operations takes $O(V \cdot L)$ per step. By indexing all vocabulary tokens in an in-memory prefix Trie during startup, prefix lookups execute in $O(K)$ time (where $K$ is the target prefix length), keeping token selection under 0.1 ms.

Byte-Pair Unicode Re-mapping: BPE tokenizers use special Unicode shift mappings for raw byte values (e.g., whitespace and control symbols). We implemented _bytes_to_unicode_mapping to ensure bidirectional consistency between raw vocab strings and decoded UTF-8 text.

Vectorized NumPy Masking: Rather than filtering logits with Python loops, mask application utilizes contiguous C-level memory buffers (np.full with -np.inf and index-slice assignments), preventing CPU bottlenecks.

Native ChatML Prompt Formatting: Leveraging Qwen's pre-trained chat format (<|im_start|>...<|im_end|>) maximizes the SLM's natural semantic understanding of tool signatures prior to decoding constraints.

## Challenges Faced
### Number Repetition Traps: 
Early implementations allowed the model to emit digits continuously during number parsing. Because digit tokens have strong self-attention affinity, the SLM entered infinite repetition loops (e.g., 200000...). I resolved this by unlocking trailing delimiters (commas , and closing braces }) as valid choices immediately after the first digit was generated.

### Compound Quote Delimiters: 
In JSON strings, the model often predicts compound tokens like ", (quote followed by comma) or "}}. Restricting string endings to isolated quotes caused loop deadlocks. I updated the FSM to recognize and parse composite closing delimiters while updating buffer pointers accordingly.

## Resources

    Qwen2.5 / Qwen3 Model Architecture and ChatML Specification

    Outlines: Fast and Deterministic Structured Generation

    NumPy Documentation on Vectorized Array Operations

## AI Usage Disclosure

Artificial Intelligence (Google Gemini) was utilized during development as an interactive thought partner and technical peer. Specifically, AI assisted in:

    Diagnosing logit distribution dead-ends and repetition traps in greedy decoding.

    Designing state transitions for the JSON Finite State Machine.

    Structuring technical documentation according to 42 evaluation standards.

    Everything was done, reviewed, and coded by me, solely with the help of a different "peer" talking to me.