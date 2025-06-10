from typing import Optional

import torch.nn as nn


class HuggingfaceToTensorWrapper(nn.Module):
    """
    A lightweight wrapper to adapt Hugging Face transformer models
    for compatibility with tools like pytorch-gradcam, which expect
    a standard PyTorch `nn.Module` that outputs a tensor.

    This wrapper extracts the `logits` from the Hugging Face model output,
    which is typically a `ModelOutput` object or a named tuple.

    Example:
        >>> from transformers import BertForSequenceClassification
        >>> model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> wrapped_model = HuggingfaceToTensorWrapper(model)
        >>> output = wrapped_model(input_tensor)  # output is a tensor (logits)

    Args:
        model (nn.Module): A Hugging Face transformer model that returns a
                           `ModelOutput` object containing `logits`.

    Forward Args:
        x (torch.Tensor or dict): Input to the underlying Hugging Face model.
                                  Typically a dict of tensors for text models
                                  (e.g., {'input_ids': ..., 'attention_mask': ...}).

    Returns:
        torch.Tensor: The `logits` tensor from the Hugging Face model output.
    """

    def __init__(self, model, device: Optional[str] = None):
        super(HuggingfaceToTensorWrapper, self).__init__()
        self.model = model

        if device:
            self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x).logits
