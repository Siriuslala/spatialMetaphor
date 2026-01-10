import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Union
from transformer_lens import HookedTransformer


class EAPDataset(Dataset):
    """
    Dataset for EAP (edge attribution patching)
    """
    def __init__(
        self,
        model: HookedTransformer,
        tokenizer: Any,
        clean_prompts: List[str],
        corrupt_prompts: List[str],
        target_clean_tokens: List[str],
        target_corrupt_tokens: List[str],
        device: str = "cpu"
    ):
        self.tokenizer = model.tokenizer if tokenizer is None else tokenizer
        self.device = device
        self.num_examples = len(clean_prompts)

        if len(clean_prompts) != len(corrupt_prompts):
            raise ValueError("The length of clean_prompts and corrupt_prompts must be the same.")
        
        # 1. Get input_ids and attention_mask
        clean_inputs_encoded = self.tokenizer.batch_encode_plus(
            clean_prompts, padding=True, return_tensors='pt'
        )
        corr_inputs_encoded = self.tokenizer.batch_encode_plus(
            corrupt_prompts, padding=True, return_tensors='pt'
        )

        self.clean_input_ids = clean_inputs_encoded["input_ids"].to(self.device)
        self.clean_attention_mask = clean_inputs_encoded["attention_mask"].to(self.device)
        self.corr_input_ids = corr_inputs_encoded["input_ids"].to(self.device)
        self.corr_attention_mask = corr_inputs_encoded["attention_mask"].to(self.device)
        
        # 2. Get end_positions and token_ids
        self.clean_end_positions = []
        self.corr_end_positions = []
        self.clean_token_ids = []
        self.corr_token_ids = []
        
        for i in range(self.num_examples):
            clean_tokens = self.tokenizer.tokenize(clean_prompts[i])
            corr_tokens = self.tokenizer.tokenize(corrupt_prompts[i])
            
            target_clean_str = "Ġ" + target_clean_tokens[i] if target_clean_tokens[i].isalnum() else target_clean_tokens[i]
            target_corr_str = "Ġ" + target_corrupt_tokens[i] if target_corrupt_tokens[i].isalnum() else target_corrupt_tokens[i]
            
            try:
                clean_end_pos = self.get_target_token_position(clean_tokens, target_clean_str)
                corr_end_pos = self.get_target_token_position(corr_tokens, target_corr_str)
                self.clean_end_positions.append(clean_end_pos)
                self.corr_end_positions.append(corr_end_pos)
                
                self.clean_token_ids.append(self.tokenizer.convert_tokens_to_ids(target_clean_str))
                self.corr_token_ids.append(self.tokenizer.convert_tokens_to_ids(target_corr_str))
                
            except ValueError as e:
                print(f"Warning: Could not find target token in prompt {i}. Please check your prompts and target token lists. Error: {e}")
                raise
                
    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "clean_inputs": {
                "input_ids": self.clean_input_ids[idx],
                "attention_mask": self.clean_attention_mask[idx],
                "end_positions": self.clean_end_positions[idx],
                "clean_token_ids": self.clean_token_ids[idx],
            },
            "corrupted_inputs": {
                "input_ids": self.corr_input_ids[idx],
                "attention_mask": self.corr_attention_mask[idx],
                "end_positions": self.corr_end_positions[idx],
                "corrupted_token_ids": self.corr_token_ids[idx],
            },
        }
    
    def get_target_token_position(self, tokens: List[str], target_token: str) -> int:
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == target_token:
                return i
        raise ValueError(f"Target token {target_token} not found in tokens {tokens}")

    def get_full_batch(self) -> Dict[str, Dict[str, Union[torch.Tensor, List[int]]]]:
        """
        Return the full batch of the dataset as a dictionary.
        """
        return {
            "clean_inputs": {
                "input_ids": self.clean_input_ids,
                "attention_mask": self.clean_attention_mask,
                "end_positions": self.clean_end_positions,
                "clean_token_ids": self.clean_token_ids,
            },
            "corrupted_inputs": {
                "input_ids": self.corr_input_ids,
                "attention_mask": self.corr_attention_mask,
                "end_positions": self.corr_end_positions,
                "corrupted_token_ids": self.corr_token_ids,
            },
        }


if __name__ == "__main__":
    
    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = model.tokenizer
    clean_prompts = ["He just won the game and is feeling happy."]
    corrupt_prompts = ["He just lost the game and is feeling sad."]
    target_clean_tokens = ["happy"]
    target_corrupt_tokens = ["sad"]
    # dataset = EAPDataset(
    #     model=model,
    #     tokenizer=tokenizer,
    #     clean_prompts=clean_prompts,
    #     corrupt_prompts=corrupt_prompts,
    #     target_clean_tokens=target_clean_tokens,
    #     target_corrupt_tokens=target_corrupt_tokens,
    # )
    # batch = dataset.get_full_batch()
    # print(batch)

    print(tokenizer.tokenize("He just won the game and is feeling happy."))
    print(tokenizer.tokenize("happy"))