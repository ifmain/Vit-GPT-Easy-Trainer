# gptvit_ezTrain.py

A script for training a GPT2-ViT (Vision Transformer with GPT2) model on custom datasets with configurable training parameters using command line arguments. This script utilizes the HuggingFace Transformers library and PyTorch for efficient processing and training.

## Features

- Load and preprocess image and text data from a specified directory.
- Train the GPT2-ViT model with configurable training parameters.
- Split large datasets into smaller subgroups for training on GPUs with limited RAM.
- Save the trained model, tokenizer, and processor for future use.

## Usage

Run the script with the following command:

```sh
python gptvit_ezTrain.py --data_dir /path/to/data --path_to_model nlpconnect/vit-gpt2-image-captioning --subgroups_count 4 --save_dir ./results --learning_rate 2e-5 --num_train_epochs 5
```

# Example tree

```
project-root/
├── data/
│   ├── image1/
│   │   ├── image1.jpeg
│   │   ├── image1.txt
│   ├── image2/
│   │   ├── image2.jpeg
│   │   ├── image2.txt
│   ├── image3/
│   │   ├── image3.jpeg
│   │   ├── image3.txt
│   ├── image4/
│   │   ├── image4.jpeg
│   │   ├── image4.txt
```

### Command Line Arguments

- `--data_dir` (str): Directory containing the dataset (required).
- `--path_to_model` (str): Path to the pretrained GPT2-ViT model (default: `nlpconnect/vit-gpt2-image-captioning`).
- `--subgroups_count` (int): Number of subgroups to split the dataset for training (default: `4`).
- `--save_dir` (str): Directory that will contain the final model file (required).
- `--output_dir` (str): Directory to save the results (default: `./results`).
- `--learning_rate` (float): Learning rate for training (default: `2e-5`).
- `--per_device_train_batch_size` (int): Batch size per device during training (default: `1`).
- `--per_device_eval_batch_size` (int): Batch size per device during evaluation (default: `1`).
- `--num_train_epochs` (int): Number of training epochs (default: `5`).
- `--weight_decay` (float): Weight decay for optimization (default: `0.01`).
- `--logging_dir` (str): Directory for logging (default: `./logs`).
- `--logging_steps` (int): Logging steps (default: `10`).
- `--save_total_limit` (int): Limit the total number of checkpoints (default: `2`).
- `--save_steps` (int): Save checkpoint every X updates steps (default: `5000`).
- `--remove_unused_columns` (bool): Remove unused columns (default: `False`).

## Contributing

Feel free to submit issues and enhancement requests.
