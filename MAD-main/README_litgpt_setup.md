# MAD with litgpt Models Setup

This setup allows you to run the MAD (Multi-Agent Debate) benchmark using three Qwen models served via litgpt on different ports.

## Prerequisites

1. **litgpt installed**: Make sure you have litgpt installed and working
2. **Qwen model checkpoints**: Download the Qwen2.5-7B-Instruct model to `checkpoints/Qwen/Qwen2.5-7B-Instruct/`
3. **Python dependencies**: Install required packages (`requests`, `datasets`, etc.)

## Quick Start

### 1. Launch the Models

```bash
# Launch all three models
python launch_models.py
```

This will start:
- **Agent A** on port 8000
- **Agent B** on port 8001  
- **Judge** on port 8003

### 2. Test the Setup

```bash
# In another terminal, test the MMLU Professional Medicine dataset
python test_mmlu_pro_med.py
```

### 3. Run the Full Benchmark

```bash
# Run the complete MAD benchmark
python src/runners/run_benchmark.py \
  --benchmark configs/benchmark.yaml \
  --models configs/models.yaml \
  --datasets configs/datasets.yaml \
  --prompts configs/prompts.yaml
```

## Model Configuration

The models are configured as follows:

| Model | Port | Purpose | Temperature | Max Tokens |
|-------|------|---------|-------------|------------|
| Agent A | 8000 | Primary debater | 0.7 | 1024 |
| Agent B | 8001 | Opposing debater | 0.8 | 1024 |
| Judge | 8003 | Debate evaluation | 0.2 | 2048 |

## Manual Model Launch

If you prefer to launch models manually:

```bash
# Terminal 1: Agent A
litgpt serve checkpoints/Qwen/Qwen2.5-3B-Instruct --port 8000

# Terminal 2: Agent B  
litgpt serve checkpoints/Qwen/Qwen2.5-3B-Instruct --port 8001

# Terminal 3: Judge
litgpt serve checkpoints/Qwen/Qwen2.5-3B-Instruct --port 8003
```

## Troubleshooting

### Models not accessible
- Check if litgpt is running on the correct ports
- Verify firewall settings allow connections to ports 8000, 8001, 8003
- Ensure the model checkpoints exist and are valid

### Port conflicts
- Change ports in `launch_models.py` and `test_mmlu_pro_med.py`
- Make sure no other services are using the specified ports

### Memory issues
- The 3B model should work on most systems
- If using GPU, ensure sufficient VRAM
- Consider using CPU-only mode if needed

## Stopping Models

- **Automatic**: Press `Ctrl+C` in the `launch_models.py` terminal
- **Manual**: Kill the processes or stop each litgpt instance

## Benefits of this Setup

1. **Resource efficiency**: Models are loaded once and reused
2. **Scalability**: Easy to add more models or change configurations
3. **Debugging**: Can test individual models independently
4. **Production ready**: Suitable for running multiple experiments
