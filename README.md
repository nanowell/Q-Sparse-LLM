# Q-Sparse-LLM

Q-Sparse-LLM is an implementation of a sparse transformer architecture designed for efficient and high-performance language modeling. This project introduces sparsity and quantization techniques to the traditional transformer architecture, aiming to reduce computational costs and memory footprint while maintaining model performance.
![x3](https://github.com/user-attachments/assets/79bb0f9e-fb25-4eb0-8899-4e497f4e34b3)

## Features

- **Top-K Sparsity**: Implements a sparse activation mechanism that retains only the top K% of values in each layer.
- **Quantized Top-K Sparsity**: Extends the sparsity mechanism with 8-bit quantization for further efficiency.
- **ReLU²GLU Activation**: Uses a squared ReLU Gated Linear Unit for improved sparsity in feed-forward layers.

## TODO:
- **Compatibility with 1-bit LLMs**: Designed to be compatible with extremely quantized models like BitNet b1.58.

## Architecture Overview

The Q-Sparse architecture is based on the Transformer architecture with modifications to enable sparsity in the activations:

1. **Top-K Sparsity**: 
   - Applies a mask to keep only the top K% of activations (by magnitude).
   - Rescales the output by its L2 norm.

2. **Quantized Top-K Sparsity**:
   - Quantizes the input to 8-bit representation before applying Top-K sparsity.

3. **Squared ReLU (ReLU²GLU)**:
   - Implements ReLU²GLU for feed-forward layers: `ReLU²GLU(X) = X · W_up^T ⊙ ReLU²(X · W_gate^T)`

## Experiment: ReLU vs ReLU2GLU
# ReLU
![image](https://github.com/user-attachments/assets/6fb08565-6e50-4262-a755-84965d684682)

# ReLU2GLU
![image](https://github.com/user-attachments/assets/f159cf4d-fe4d-4b16-b87a-2cc3dcd14104)


## Installation

```bash
git clone https://github.com/nanowell/Q-Sparse-LLM.git
cd Q-Sparse-LLM
```

## Usage

Here's a basic example of how to use the Q-Sparse-LLM model:

```python
from q_sparse_llm import QSparseModel

# Initialize the model
model = QSparseModel(
    vocab_size=30000,
    d_model=768,
    nhead=12,
    num_layers=12,
    dim_feedforward=3072,
    k_ratio=0.5,
    quantized=True
)

# Use the model for inference or training
# (Add specific usage instructions based on your implementation)
```

## Contributing

Contributions to Q-Sparse-LLM are welcome!

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use Q-Sparse-LLM in your research, please cite:

```
@software{Q-Sparse-LLM,
  author = {nanowell},
  title = {Q-Sparse-LLM: Quantized Sparse Language Model},
  year = {2024},
  url = {https://github.com/nanowell/Q-Sparse-LLM}
}
```

## Acknowledgements

This project builds upon the work Q-Sparse paper:
```
@misc{wang2024qsparselargelanguagemodels,
      title={Q-Sparse: All Large Language Models can be Fully Sparsely-Activated}, 
      author={Hongyu Wang and Shuming Ma and Ruiping Wang and Furu Wei},
      year={2024},
      eprint={2407.10969},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.10969}, 
}
```
## Contact

For questions and feedback, please open an issue in the GitHub repository or contact [zarugeos@gmail.com].
