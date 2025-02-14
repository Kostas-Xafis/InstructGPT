# InstructGPT Project

## Table of Contents

- [InstructGPT Project](#instructgpt-project)
  - [Table of Contents](#table-of-contents)
  - [Summary](#summary)
  - [Installation](#installation)
  - [Usage](#usage)

## Summary

This project demonstrates the use of large language models (LLMs) for different tasks including simple prompting, diverse interaction scenarios, generating alternative responses, and embedding similarity analysis. The project integrates functionalities from Llama and Hugging Face's model hub to download models, process chat interactions, and compute embedding similarities.

## Installation

1. Clone the repository.

   ```bash
   git clone https://github.com/Kostas-Xafis/InstructGPT.git
   ```

2. Create a Python virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the environment:

    ```bash
    source venv/bin/activate
    ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the project with different modes by specifying the mode argument:

```bash
python llm.py --mode 1
```

Modes:

- `1`: Simple Questions
- `2`: Variety of Questions
- `3`: Alternative Responses with various temperature and top-p configurations
- `4`: Embedding Similarity (bonus exercise)

Use `--log` to enable logging of assistant responses to the `logs/` directory:

```bash
python llm.py --mode 1 --log True
```
