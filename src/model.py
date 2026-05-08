import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from config import DEVICE, MODEL_NAME, HOOK_NAME, SAE_RELEASE, SAE_ID


def load_model() -> HookedTransformer:
    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()
    print(f"  {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def extract_activations(
    model: HookedTransformer,
    items: list,
) -> tuple[torch.Tensor, list[str]]:
    """
    Tokenise each prompt and return the residual-stream activation at LAYER
    for the LAST token position (the historical concept token).

    Returns
    -------
    activations : (N, d_model) CPU tensor
    labels      : list of N display labels
    """
    activations, labels = [], []
    for label, prompt, *_ in tqdm(items, desc="  activations", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=HOOK_NAME, device=DEVICE
            )
        activations.append(cache[HOOK_NAME][0, -1, :].cpu())
        labels.append(label)
    return torch.stack(activations), labels


