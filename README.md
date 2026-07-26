<div align="center">

# TuneKit

### The fine-tuning workflow you wish existed.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Powered by Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-FF6F00.svg)](https://github.com/unslothai/unsloth)
[![Free to Use](https://img.shields.io/badge/Cost-$0-22c55e.svg)](https://tunekit.app)

[**Try it now**](https://tunekit.app) · [Report Bug](https://github.com/riyanshibohra/TuneKit/issues) · [Request Feature](https://github.com/riyanshibohra/TuneKit/issues)

P.S. TuneKit trending at **#19** on Product Hunt (Day of Launch) with 88 followers & 50+ Upvotes - [check it out](https://www.producthunt.com/products/tunekit?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-tunekit)!

<video src="https://github.com/user-attachments/assets/ce06d4bc-7a74-4eef-8862-27d0f74aef00" controls autoplay muted loop></video>

*See it in action*

</div>

---

## The Problem

Fine-tuning LLMs is powerful but painful. You need to:
- Set up CUDA, PyTorch, and a dozen dependencies
- Rent expensive GPUs or wait hours on slow hardware
- Write training scripts, manage hyperparameters, handle OOM errors
- Figure out how to export and deploy your model

**Most developers give up before they start.**

## The Solution

TuneKit eliminates all of that. Upload your data, answer two questions, and get a ready-to-run Colab notebook. Click "Run All" and your fine-tuned model is ready in ~15 minutes.

```
Your Data → TuneKit → Colab Notebook → Fine-tuned Model
```

No local GPU. No dependencies. No cost (uses Google's free T4).

---

## How It Works

| Step | What You Do | What TuneKit Does |
|:----:|-------------|-------------------|
| **1** | Upload JSONL file | Validates format, analyzes patterns |
| **2** | Answer 2 questions | Recommends optimal model + hyperparameters |
| **3** | Click "Get Notebook" | Generates pre-configured Colab notebook |
| **4** | Hit "Run All" in Colab | Trains on free T4 GPU (~15 min) |
| **5** | Download your model | Export as LoRA, GGUF, or merged weights |

---

## Why TuneKit?

<table>
<tr>
<td width="50%">

### Without TuneKit
- Hours of environment setup
- $50-200 in GPU costs
- Hundreds of lines of code
- Trial and error with hyperparameters
- OOM errors and debugging

</td>
<td width="50%">

### With TuneKit
- Zero setup
- $0 (free Colab GPU)
- Zero code to write
- AI-optimized configuration
- Just works

</td>
</tr>
</table>

**Powered by [Unsloth](https://github.com/unslothai/unsloth)** - 2x faster training, 70% less VRAM.

---

## Supported Models

| Model | Parameters | Best For |
|-------|------------|----------|
| **Phi-4 Mini** | 3.8B | Classification, extraction, structured output |
| **Llama 3.2** | 1B, 3B | Q&A, conversational AI, context tracking |
| **Mistral 7B** | 7B | Long-form generation, complex reasoning |
| **Qwen 2.5** | 1.5B, 3B | Multilingual, JSON output, structured data |
| **Gemma 2** | 2B | Edge deployment, mobile, fast inference |

---

## Data Format

TuneKit uses the standard conversation format:

```jsonl
{"messages": [{"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris is the capital of France."}]}
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write hello world in Python"}, {"role": "assistant", "content": "print('Hello, World!')"}]}
```

**Requirements:**
- JSONL format (one JSON object per line)
- Each line has a `messages` array
- Messages have `role` (user/assistant/system) and `content`
- Minimum 50 examples (100-1000 recommended)

---

## Data Enrichment

TuneKit now includes tools to automatically improve your dataset quality before training:

- **Quality Scoring**: Evaluates every conversation on complexity, lexical diversity, and dialogue balance.
- **Smart Prioritization**: Automatically ranks examples and filters out low-quality ones.
- **Class Balancing**: Detects underrepresented classes in classification datasets and automatically balances them.

---

## Development Setup

### Prerequisites
- Python 3.10+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/riyanshibohra/TuneKit.git
   cd TuneKit
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**
   Copy `.env.example` to `.env` and configure your tokens:
   ```bash
   # Windows (PowerShell)
   cp .env.example .env

   # macOS/Linux
   cp .env.example .env
   ```
   
   > **Note:** A `GITHUB_TOKEN` is required to automatically create private Gists for the Colab notebooks.
   > [Generate a token here](https://github.com/settings/tokens) (Scope: `gist`).

5. **Start the server**
   ```bash
   uvicorn api.main:app --reload
   ```

   The app will be available at [http://localhost:8000](http://localhost:8000).

---

## Tech Stack

- **Frontend:** Vanilla JS, CSS
- **Backend:** FastAPI, Python
- **Training:** Unsloth, Transformers, PEFT
- **Infrastructure:** Google Colab (free T4 GPU)

---

## Contributing

Contributions are welcome! Feel free to:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - use it for anything.

---

<div align="center">

**[tunekit.app](https://tunekit.app)**

Built with lots of caffeine + curiosity.

</div>
