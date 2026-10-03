"""ML2025 Spring HW3 - Understanding LLM / Transformers (local PyTorch version).

Ported from the official Colab:
https://colab.research.google.com/drive/1Ku_p27ml8QJ-Rd7FilZQvDA9axe68V7o

Differences from Colab:
- HF token is read from `HF_TOKEN` / `hf auth login` cache instead of being hard-coded.
- Figures are saved to `HW03/outputs/` instead of `plt.show()`.
- Q2 replays the 3 conversation turns from the HW3 slides (use `--interactive` to chat).
- Each question can be run on its own: `python HW03/hw3.py --q 1 3 4`.
Everything else (including quirks like the double <bos> in Q1/Q2) is kept identical to
the Colab so results match the TA reference answers.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HybridCache,
)

MODEL_ID = "google/gemma-2-2b-it"
SCORING_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda"
DTYPE = torch.float16
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def save_fig(name):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"[saved] {path}")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        torch_dtype=DTYPE,
        # Gemma2 uses eager attention by default in transformers 4.47; Q6 needs attention weights
        attn_implementation="eager",
    )
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Q1: Chat template comparison
# ---------------------------------------------------------------------------
def calculate_coherence(question, answer, scoring_model, scoring_tokenizer):
    features = scoring_tokenizer([question], [answer], padding=True, truncation=True, return_tensors="pt")
    scoring_model.eval()
    with torch.no_grad():
        scores = scoring_model(**features).logits.squeeze().item()
    return scores


def generate_text_from_prompt(prompt, tokenizer, model, max_new_tokens):
    print("========== Prompt inputted to the model ==========\n", prompt)

    # Tokenize the prompt (same as Colab: a chat-template prompt ends up with two <bos>)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    ######################## TODO (Q1.1 ~ 1.4) ########################
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    ###################################################################
    if output_ids is not None and len(output_ids) > 0:
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        return "Empty Response"


def q1(tokenizer, model, args):
    scoring_model = AutoModelForSequenceClassification.from_pretrained(SCORING_MODEL_ID)
    scoring_tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL_ID)

    question = "Please tell me about the key differences between supervised learning and unsupervised learning. Answer in 200 words."

    # With chat template
    chat = [{"role": "user", "content": question}]
    prompt_with_template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    response_with_template = generate_text_from_prompt(prompt_with_template, tokenizer, model, args.max_new_tokens)
    response_with_template = response_with_template.split("model\n")[-1].strip("\n").strip()
    print("========== Output ==========\n", response_with_template)
    score_with = calculate_coherence(question, response_with_template, scoring_model, scoring_tokenizer)
    print(f"========== Coherence Score (with template) : {score_with:.4f}  ==========\n")

    # Without chat template (directly using plain text)
    response_without_template = generate_text_from_prompt(question, tokenizer, model, args.max_new_tokens)
    response_without_template = response_without_template.split(question.split(" ")[-1])[-1].strip("\n").strip()
    print("========== Output ==========\n", response_without_template)
    score_without = calculate_coherence(question, response_without_template, scoring_model, scoring_tokenizer)
    print(f"========== Coherence Score (without template) : {score_without:.4f}  ==========")


# ---------------------------------------------------------------------------
# Q2: Multi-turn conversations
# ---------------------------------------------------------------------------
Q2_TURNS = [
    "Name a color in a rainbow, please just answer in a word without any emoji.",
    "That's great! Now, could you tell me another color that I can find in a rainbow?",
    "Could you continue and name yet another color from the rainbow?",
]


def q2(tokenizer, model, args):
    if args.interactive:
        print("Chatbot: Hello! How can I assist you today? (Type 'exit' to quit)")
        turns = iter(lambda: input("You: "), "exit")
    else:
        turns = Q2_TURNS

    chat_history = []
    for round_idx, user_input in enumerate(turns, start=1):
        if not args.interactive:
            print(f"You: {user_input}")
        chat_history.append({"role": "user", "content": user_input})
        chat_template_format_prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        ######################## (Q2.1 ~ 2.3) ########################
        print(f"=== Prompt with chat template format inputted to the model on round {round_idx} ===\n{chat_template_format_prompt}")
        print("===============================================")
        ###################################################################

        inputs = tokenizer(chat_template_format_prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs_p = model(**inputs)

        last_token_logits = outputs_p.logits[:, -1, :]
        probs = torch.nn.functional.softmax(last_token_logits.float(), dim=-1)

        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs.cpu().squeeze().numpy()
        top_indices = top_indices.cpu().squeeze().numpy()
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]
        print("Top-10 next tokens:")
        for t, p in zip(top_tokens, top_probs):
            print(f"  {t!r:>16}  {p:.4f}")

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_probs, y=top_tokens, hue=top_tokens, palette="coolwarm", legend=False)
        plt.xlabel("Probability")
        plt.ylabel("Token")
        plt.title(f"Top Token Probabilities for Next Word (round {round_idx})")
        save_fig(f"q2_round{round_idx}_top_tokens.png")

        outputs = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"Chatbot: {response}\n")
        chat_history.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Q3: Tokenization of a sentence
# ---------------------------------------------------------------------------
def q3(tokenizer, model, args):
    sentence = args.sentence

    ######################## TODO (Q3.1 ~ 3.4) ########################
    # Encode the sentence into token IDs without adding special tokens
    token_ids = tokenizer.encode(sentence, add_special_tokens=False)
    # Convert the token IDs back to their corresponding tokens (words or subwords)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    ###################################################################

    for t, t_id in zip(tokens, token_ids):
        print(f"Token: {t}, token index: {t_id}")


# ---------------------------------------------------------------------------
# Q4: Auto-regressive generation
# ---------------------------------------------------------------------------
def compute_self_bleu(generated_sentences):
    from nltk.translate.bleu_score import sentence_bleu

    total_bleu_score = 0
    num_sentences = len(generated_sentences)
    for i, hypothesis in enumerate(generated_sentences):
        references = [generated_sentences[j] for j in range(num_sentences) if j != i]
        bleu_scores = [sentence_bleu([ref.split()], hypothesis.split()) for ref in references]
        total_bleu_score += sum(bleu_scores) / len(bleu_scores)
    return total_bleu_score / num_sentences


def q4(tokenizer, model, args):
    max_generation_tokens = 30

    ######################## TODO (Q4.3 ~ 4.6) ########################
    top_k = args.top_k  # Set K for top-k sampling
    top_p = args.top_p  # Set P for nucleus sampling
    ###################################################################

    prompt = "Generate a paraphrase of the sentence 'Professor Hung-yi Lee is one of the best teachers in the domain of machine learning'. Just response with one sentence."
    input_ids = tokenizer(prompt, return_tensors="pt")

    generation_params = {
        "do_sample": True,
        "max_length": max_generation_tokens + len(input_ids.input_ids[0]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "attention_mask": input_ids.attention_mask.to(DEVICE),
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }

    generated_sentences_top_k = []
    generated_sentences_top_p = []
    for method in ["top-k", "top-p"]:
        for _ in trange(args.num_samples, desc=method):
            if method == "top-k":
                generated_output = model.generate(input_ids=input_ids.input_ids.to(DEVICE), top_k=top_k, **generation_params)
            else:
                ######################## TODO (Q4.3 ~ 4.6) ########################
                generated_output = model.generate(input_ids=input_ids.input_ids.to(DEVICE), top_p=top_p, **generation_params)
                ###################################################################
            generated_tokens = generated_output.sequences[0, len(input_ids.input_ids[0]):]
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            sentence = decoded_text.replace(" ,", ",").replace(" 's", "'s").replace(" .", ".").strip()
            if method == "top-k":
                generated_sentences_top_k.append(sentence)
            else:
                generated_sentences_top_p.append(sentence)

    print("===== Top-K Sampling Output =====\n")
    for idx, sentence in enumerate(generated_sentences_top_k):
        print(f"{idx}. {sentence}")
    print("\n===== Top-P Sampling Output =====\n")
    for idx, sentence in enumerate(generated_sentences_top_p):
        print(f"{idx}. {sentence}")
    print()

    print(f"self-BLEU Score for top_k (k={top_k}): {compute_self_bleu(generated_sentences_top_k):.4f}")
    print(f"self-BLEU Score for top_p (p={top_p}): {compute_self_bleu(generated_sentences_top_p):.4f}")


# ---------------------------------------------------------------------------
# Q5: t-SNE
# ---------------------------------------------------------------------------
def q5(tokenizer, model, args):
    from sklearn.manifold import TSNE

    ######################## (Q5.2 ~ 5.3) ########################
    sentences = [
        "I ate a fresh apple.",  # Apple (fruit)
        "Apple released the new iPhone.",  # Apple (company)
        "I peeled an orange and ate it.",  # Orange (fruit)
        "The Orange network has great coverage.",  # Orange (telecom)
        "Microsoft announced a new update.",  # Microsoft (company)
        "Banana is my favorite fruit.",  # Banana (fruit)
    ]

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1]  # Extract last layer embeddings
    # Compute sentence-level embeddings (mean pooling, padding included as in the Colab)
    sentence_embeddings = hidden_states.mean(dim=1).float().cpu().numpy()

    word_labels = [
        "Apple (fruit)", "Apple (company)",
        "Orange (fruit)", "Orange (telecom)",
        "Microsoft (company)", "Banana (fruit)",
    ]

    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(sentence_embeddings)

    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "orange", "purple", "green", "brown"]
    for i, label in enumerate(word_labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=colors[i], s=100)
        plt.text(embeddings_2d[i, 0] + 0.1, embeddings_2d[i, 1] + 0.1, label, fontsize=12, color=colors[i])
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.title("t-SNE Visualization of Word Embeddings")
    save_fig("q5_tsne.png")
    ##################################################


# ---------------------------------------------------------------------------
# Q6: Attention weights
# ---------------------------------------------------------------------------
def plot_attention(attn_matrix, tokens, title, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis", annot=False)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    save_fig(filename)


def q6(tokenizer, model, args):
    prompt = "Google "
    input_ids = tokenizer(prompt, return_tensors="pt")
    next_token_id = input_ids.input_ids.to(DEVICE)
    attention_mask = input_ids.attention_mask.to(DEVICE)
    cache_position = torch.arange(attention_mask.shape[1], device=DEVICE)

    generation_tokens = 20
    total_tokens = generation_tokens + next_token_id.size(1) - 1
    layer_idx = args.layer_idx
    head_idx = args.head_idx

    kv_cache = HybridCache(config=model.config, max_batch_size=1, max_cache_len=total_tokens, device=DEVICE, dtype=DTYPE)

    generated_tokens = []
    attentions = None
    for num_new_tokens in range(generation_tokens):
        with torch.no_grad():
            outputs = model(
                next_token_id,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True,
            )

        ######################## TODO (Q6.1 ~ 6.4) ########################
        # Get the logits for the last generated token from outputs
        logits = outputs.logits[:, -1, :]
        # Extract the attention scores from the model's outputs
        attention_scores = outputs.attentions
        ###################################################################

        last_layer_attention = attention_scores[layer_idx][0][head_idx].detach().float().cpu().numpy()
        if num_new_tokens == 0:
            attentions = last_layer_attention
        else:
            attentions = np.append(attentions, last_layer_attention, axis=0)

        next_token_id = logits.argmax(dim=-1)
        generated_tokens.append(next_token_id.item())

        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=DEVICE)], dim=-1)
        next_token_id = next_token_id.unsqueeze(0)
        kv_cache = outputs.past_key_values
        cache_position = cache_position[-1:] + 1

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    print(f"Generated text: {full_text!r}")
    print(f"Attention matrix shape: {attentions.shape}")

    tokens = tokenizer.tokenize(full_text)
    plot_attention(attentions, tokens, f"Attention Weights for Generated Token of Layer {layer_idx}", f"q6_attention_L{layer_idx}_H{head_idx}.png")


# ---------------------------------------------------------------------------
# Q7: SAE activations (Gemma Scope)
# ---------------------------------------------------------------------------
def load_sae():
    from sae_lens import SAE

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_20/width_16k/canonical",
    )
    print(sae, cfg_dict, sparsity)
    return sae.to(DEVICE)


def get_dashboard_html(sae_release="gemma-2-2b", sae_id="20-gemmascope-res-16k", feature_idx=0):
    html_template = "https://neuronpedia.org/{}/{}/{}?embed=true&embedexplanation=true&embedplots=true&embedtest=true&height=300"
    return html_template.format(sae_release, sae_id, feature_idx)


@torch.no_grad()
def get_max_activation(model, tokenizer, sae, prompt, feature_idx=10004, tag=""):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    outputs = model(tokens, output_hidden_states=True)

    hidden_states = outputs.hidden_states[sae.cfg.hook_layer]
    feature_acts = sae.encode(hidden_states).squeeze()
    feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1])

    max_activation = feature_acts[:, feature_idx].max().item()

    plt.figure(figsize=(8, 5))
    plt.hist(feature_acts[:, feature_idx].float().cpu().numpy(), bins=50, alpha=0.75, color="blue", edgecolor="black")
    plt.xlabel(f"Activation values (Feature {feature_idx})")
    plt.ylabel("Frequency")
    plt.title(f"Activation Distribution for Feature {feature_idx} - Prompt: '{prompt}'", wrap=True)
    plt.grid(True)
    save_fig(f"q7_max_activation_{tag}.png")
    return max_activation


@torch.no_grad()
def plot_token_activations(model, tokenizer, sae, prompt, feature_idx=10004, layer_idx=0):
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    token_list = tokenizer.convert_ids_to_tokens(token_ids.squeeze().tolist())
    outputs = model(token_ids, output_hidden_states=True)

    layer_idx = layer_idx if layer_idx is not None else sae.cfg.hook_layer
    feature_acts = sae.encode(outputs.hidden_states[layer_idx]).squeeze()
    print(f"feature_acts shape: {feature_acts.shape}")
    activations = feature_acts[:, feature_idx].squeeze().float().cpu().numpy()
    for t, a in zip(token_list, activations):
        print(f"  {t!r:>16}  {a:.4f}")

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(token_list)), activations, color="blue", alpha=0.7)
    plt.xticks(range(len(token_list)), token_list, rotation=45)
    plt.xlabel("Tokens")
    plt.ylabel(f"Activation Value (Feature {feature_idx})")
    plt.title(f"Token-wise Activations for Layer {layer_idx}")
    plt.grid(True)
    save_fig(f"q7_token_activations_L{layer_idx}.png")


@torch.no_grad()
def plot_layer_activations(model, tokenizer, sae, prompt, token_idx=0, feature_idx=10004):
    token_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
    token_list = tokenizer.convert_ids_to_tokens(token_ids.squeeze().tolist())
    outputs = model(token_ids, output_hidden_states=True)

    num_layers = len(outputs.hidden_states)
    activations = []
    for layer_idx in range(num_layers):
        feature_acts = sae.encode(outputs.hidden_states[layer_idx]).squeeze()
        activations.append(feature_acts[token_idx, feature_idx].item())
    print(f"Token {token_idx} = {token_list[token_idx]!r}, activations per layer:")
    print("  " + ", ".join(f"L{i}:{a:.2f}" for i, a in enumerate(activations)))

    plt.figure(figsize=(8, 5))
    plt.plot(range(num_layers), activations, marker="o", linestyle="-", color="blue")
    plt.xlabel("Layer")
    plt.ylabel(f"Activation Value (Feature {feature_idx})")
    plt.title(f"Activation Across Layers for Token '{token_list[token_idx]}'")
    plt.xticks(range(num_layers))
    plt.grid(True)
    save_fig(f"q7_layer_activations_tok{token_idx}.png")


def q7(tokenizer, model, args):
    sae = load_sae()
    feature_idx = 10004

    ########################## TODO (Q7.1) ############################
    print(f"[Q7.1] Open in browser: {get_dashboard_html(feature_idx=feature_idx)}")
    ###################################################################

    ######################## (Q7.2 ~ 7.3) ########################
    prompt_a = "Time travel offers me the opportunity to correct past errors, but it comes with its own set of risks."
    prompt_b = "I accept that my decisions shape my future, and though mistakes are inevitable, they define who I become."
    max_activation_a = get_max_activation(model, tokenizer, sae, prompt_a, feature_idx, tag="a")
    max_activation_b = get_max_activation(model, tokenizer, sae, prompt_b, feature_idx, tag="b")
    print(f"max_activation for prompt_a: {max_activation_a}")
    print(f"max_activation for prompt_b: {max_activation_b}")

    prompt = "Time travel will become a reality as technology continues to advance."
    ######################## (Q7.4 ~ 7.6) ########################
    plot_token_activations(model, tokenizer, sae, prompt, feature_idx, args.sae_layer_idx)

    ######################## (Q7.7 ~ 7.9) ########################
    for token_idx in args.token_idx:
        plot_layer_activations(model, tokenizer, sae, prompt, token_idx, feature_idx)


QUESTIONS = {1: q1, 2: q2, 3: q3, 4: q4, 5: q5, 6: q6, 7: q7}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q", type=int, nargs="+", default=list(QUESTIONS), choices=list(QUESTIONS), help="questions to run")
    parser.add_argument("--seed", type=int, default=None, help="random seed (Colab does not set one)")
    # Q1
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Q1 generation length")
    # Q2
    parser.add_argument("--interactive", action="store_true", help="Q2: chat interactively instead of replaying the slide turns")
    # Q3
    parser.add_argument("--sentence", default="I love taking a Machine Learning course by Professor Hung-yi Lee, What about you?")
    # Q4
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--num-samples", type=int, default=20)
    # Q6
    parser.add_argument("--layer-idx", type=int, default=10)
    parser.add_argument("--head-idx", type=int, default=7)
    # Q7
    parser.add_argument("--sae-layer-idx", type=int, default=24)
    parser.add_argument("--token-idx", type=int, nargs="+", default=[1])
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login

        login(os.environ["HF_TOKEN"])

    tokenizer, model = load_model()
    for q in args.q:
        print(f"\n{'#' * 30} Q{q} {'#' * 30}")
        QUESTIONS[q](tokenizer, model, args)


if __name__ == "__main__":
    main()
