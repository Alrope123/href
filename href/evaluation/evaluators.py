from collections import defaultdict

ANNOTATOR_SUITE_DICT = {
    "href": {
        "Brainstorm": {
            "annotator": "llama3.1-70b_basic_no_reference",
            "use_human_ref": False,
        },
        "Open QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Closed QA": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Extract": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Generation": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Rewrite": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Summarize": {
            "annotator": "llama3.1-70b_basic_no_reference",
            "use_human_ref": False,
        },
        "Classify": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Fact Checking or Attributed QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Multi-Document Synthesis": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Reasoning Over Numerical Data": {
            "annotator": "llama3.1-70b_basic_w_reference",
            "use_human_ref": True,
        },
    },
    "href_7b": {
        "Brainstorm": {
            "annotator": "llama3.1_basic_no_reference",
            "use_human_ref": False,
        },
        "Open QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Closed QA": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Extract": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Generation": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Rewrite": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Summarize": {
            "annotator": "llama3.1_basic_no_reference",
            "use_human_ref": False,
        },
        "Classify": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
        "Fact Checking or Attributed QA": {
            "annotator": "bertscore",
            "use_human_ref": True,
        },
        "Multi-Document Synthesis": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Reasoning Over Numerical Data": {
            "annotator": "llama3.1_basic_w_reference",
            "use_human_ref": True,
        },
    },
    "href_new": {
        "Brainstorm": {
            "annotator": "llama3.3-70b_basic_no_reference",
            "use_human_ref": False,
        },
        "Open QA": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Closed QA": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Extract": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Generation": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Rewrite": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Summarize": {
            "annotator": "llama3.3-70b_basic_no_reference",
            "use_human_ref": False,
        },
        "Classify": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Fact Checking or Attributed QA": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
        "Multi-Document Synthesis": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        }, 
        "Reasoning Over Numerical Data": {
            "annotator": "llama3.3-70b_basic_w_reference",
            "use_human_ref": True,
        },
    },
}

ANNOTATION_REVERSE_MAP = {
    1: 2,
    2: 1,
    0: 0
}

DEFINED_ANNOTATORS = ["short", "long", "random_no_tie", "bertscore", "rouge", "contriever", "grit", "qwen", "me5"]

def bertscore(responses_1, responses_2, human_references, args):
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en")
    # calculate score
    _, _, f1s_1 = scorer.score([res['output'] for res in responses_1], [res['output'] for res in human_references])
    _, _, f1s_2 = scorer.score([res['output'] for res in responses_2], [res['output'] for res in human_references])
    annotations = []
    for res1, res2, human_ref, f1_1, f1_2 in zip(responses_1, responses_2, human_references, f1s_1, f1s_2):
        if f1_1 > f1_2:
            score = 1.0
        elif f1_1 < f1_2:
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "output_human": human_ref["output"],
            "annotator": "bert",
            "preference": score
        })
    return annotations


def rouge(model_responses, baseline_responses, human_references, args):
    from rouge import Rouge
    rouge = Rouge()

    annotations = []
    for res1, res2, human_ref in zip(model_responses, baseline_responses, human_references):
        # get rouge scores 
        if len(res1['output'].strip()) == 0:
            eval_1_rouge = [{"rouge-1": {'f': 0}}]
        else:
            eval_1_rouge = rouge.get_scores(res1['output'], human_ref['output'])
        if len(res2['output'].strip()) == 0:
            eval_2_rouge = [{"rouge-1": {'f': 0}}]
        else:
            eval_2_rouge = rouge.get_scores(res2['output'], human_ref['output'])

        eval_1_score = eval_1_rouge[0]["rouge-1"]["f"]
        eval_2_score = eval_2_rouge[0]["rouge-1"]["f"]
        if eval_1_score > eval_2_score:
            score = 1.0
        elif eval_1_score < eval_2_score:
            score = 2.0
        else:
            score = 0.0
        
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "output_human": human_ref["output"],
            "annotator": "rouge-1",
            "preference": score
        })
    return annotations


def long(model_responses, baseline_responses, human_references, args):
    annotations = []
    for res1, res2 in zip(model_responses, baseline_responses):
        if len(res1["output"]) > len(res2["output"]):
            score = 1.0
        elif len(res1["output"]) < len(res2["output"]):
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "long",
            "preference": score
        })
    return annotations


def short(model_responses, baseline_responses, human_references, args):
    annotations = []
    for res1, res2 in zip(model_responses, baseline_responses):
        if len(res1["output"]) < len(res2["output"]):
            score = 1.0
        elif len(res1["output"]) > len(res2["output"]):
            score = 2.0
        else:
            score = 0.0
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "short",
            "preference": score
        })
    return annotations


def random_no_tie(model_responses, baseline_responses, human_references, args):
    import random
    annotations = []
    for res1, res2 in zip(model_responses, baseline_responses):
        score = 1.0 if bool(random.getrandbits(1)) else 2.0
        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "short",
            "preference": score
        })
    return annotations


def perplexity(model_responses, baseline_responses, human_references, category, args):
    import os
    import json

    generator_model = model_responses[0]["generator"]
    generator_baseline = baseline_responses[0]["generator"]
    # load model generated perplexities
    model_perplexities = defaultdict(list)
    with open(os.path.join(args.perplexity_dir, generator_model, category.lower().replace(" ", "_"), "loss.jsonl"), "r") as fin:
        model_perplexities[category].extend([json.loads(line) for line in fin])
    baseline_perplexities = defaultdict(list)
    with open(os.path.join(args.perplexity_dir, generator_baseline, category.lower().replace(" ", "_"), "loss.jsonl"), "r") as fin:
        baseline_perplexities[category].extend([json.loads(line) for line in fin])
    
    annotations = []
    for res1, ppl1, res2, ppl2, res_human in zip(model_responses, model_perplexities, baseline_responses, baseline_perplexities, human_references):
        assert ppl1['reference'] == ppl2['reference'] and ppl1['reference'] == res_human['output'], "Plexities doesn't match"
        if ppl1["output"] < ppl2["output"]:
            score = 1.0
        if ppl1["output"] > ppl2["output"]:
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "annotator": "perplexity",
            "preference": score
        })

    return annotations

def embedder(responses_1, responses_2, human_references, args, model_name):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer

    if "meta-llama" in model_name:
        tokenizer_name_or_path = model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(model_name)
    elif "GritLM" in model_name:
        from gritlm import GritLM
        tokenizer  = None
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
    elif "contriever" in model_name:
        model, tokenizer, _ = contriever.src.contriever.load_retriever(model_name)
    elif "dragon" in model_name:
        tokenizer_name_or_path = model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = AutoModel.from_pretrained(model_name)
    elif _is_sentence_transformers(model_name):
        tokenizer = None
        model = SentenceTransformer(model_name)
    else:
        print(f"{model_name} is not supported!")
        raise AttributeError

    inputs_1 = [res['output'] for res in responses_1]
    inputs_2 = [res['output'] for res in responses_2]
    inputs_human = [res['output'] for res in human_references]

    # calculate score
    embeddings_1 = _embed_passages(inputs_1, model, tokenizer, model_name)
    embeddings_2 = _embed_passages(inputs_2, model, tokenizer, model_name)
    embeddings_human = _embed_passages(inputs_human, model, tokenizer, model_name)

    annotations = []
    for res1, res2, human_ref, emb1, emb2, emb_human in zip(responses_1, responses_2, human_references, embeddings_1, embeddings_2, embeddings_human):
        cos_similarity_1 = _cosine_similarity_np(emb1, emb_human)
        cos_similarity_2 = _cosine_similarity_np(emb2, emb_human)
        if cos_similarity_1 > cos_similarity_2:
            score = 1.0
        elif cos_similarity_1 < cos_similarity_2:
            score = 2.0
        else:
            score = 0.0

        annotations.append({
            "instruction": res1["instruction"],
            "output_1": res1["output"],
            "generator_1": res1["generator"],
            "output_2": res2["output"],
            "generator_2": res2["generator"],
            "output_human": human_ref["output"],
            "annotator": "bert",
            "preference": score
        })
    return annotations

def contriever(responses_1, responses_2, human_references, args):
    return embedder(responses_1, responses_2, human_references, args, 'facebook/contriever-msmarco')

def grit(responses_1, responses_2, human_references, args):
    return embedder(responses_1, responses_2, human_references, args, 'GritLM/GritLM-7B')

def qwen(responses_1, responses_2, human_references, args):
    return embedder(responses_1, responses_2, human_references, args, 'Qwen/Qwen3-Embedding-8B')

def me5(responses_1, responses_2, human_references, args):
    return embedder(responses_1, responses_2, human_references, args, 'intfloat/multilingual-e5-large-instruct')

def _embed_passages(passages, model, tokenizer, model_name, batch_size=32):
    import torch
    from tqdm import tqdm
    import numpy as np

    device = 'cuda' if torch.cuda.is_available()  else 'cpu'
    
    if _is_sentence_transformers(model_name):
        with torch.no_grad():
            if "GritLM" in model_name:
                allembeddings = model.encode(passages, batch_size=batch_size, instruction="<|embed|>\n")
            else:
                allembeddings = model.encode(passages, batch_size=batch_size)
    
    elif "meta-llama" in model_name:
        total = 0
        allembeddings = []
        batch_text = []
        tot_psgs = len(passages)

        with torch.no_grad():
            for k, p in enumerate(tqdm(passages)):
                # Prepare text for encoding
                batch_text.append(p)

                if len(batch_text) == batch_size or k == tot_psgs - 1:
                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                    output = model(**encoded_batch)  # Get model output

                    if "contriever" not in model_name:
                        hidden_states = output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)
                        attention_mask = encoded_batch["attention_mask"]  # Shape: (batch_size, seq_len)

                        seq_len = hidden_states.shape[1]  # Get sequence length (L)
                        indices = torch.arange(1, seq_len + 1, dtype=torch.float32, device=hidden_states.device)  # Token positions

                        # Zero out weights for padding tokens
                        indices = indices * attention_mask  # Multiply by mask to remove padding influence
                        weight_sum = torch.sum(indices, dim=1, keepdim=True)  # Sum of non-padding weights per passage
                        
                        # Avoid division by zero (handle all-padding cases)
                        weight_sum = torch.where(weight_sum == 0, torch.tensor(1.0, device=hidden_states.device), weight_sum)

                        # Compute normalized weights
                        weights = indices / weight_sum  # Normalize weights
                        weights = weights.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1] for broadcasting

                        # Compute final weighted embedding
                        weighted_embedding = torch.sum(weights * hidden_states, dim=1)  # Weighted sum over tokens

                        embeddings = weighted_embedding.cpu()

                    allembeddings.append(embeddings)

                    batch_text = []

        allembeddings = torch.cat(allembeddings, dim=0).numpy()

    else:
        total = 0
        allembeddings = []
        batch_text = []
        tot_psgs = len(passages)
        with torch.no_grad():
            for k, p in enumerate(tqdm(passages)):
                batch_text.append(p)

                if len(batch_text) == batch_size or k == tot_psgs - 1:
                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                    embeddings = model(**encoded_batch)  # shape: (per_gpu_batch_size, hidden_size)
                    # if "contriever" not in model_name:
                    #     # assume in hf form
                    #     embeddings = embeddings.last_hidden_state[:, 0, :]
                    embeddings = embeddings.last_hidden_state[:, 0, :]

                    print(embeddings.shape)
                    embeddings = embeddings.cpu()
                    
                    allembeddings.append(embeddings)

                    batch_text = []
        
        allembeddings = torch.cat(allembeddings, dim=0).numpy()

    allembeddings = allembeddings.astype(np.float16)
    return allembeddings

def _cosine_similarity_np(vec1, vec2):
    import numpy as np
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def _is_sentence_transformers(model_name_or_path):
    return "sentence-transformers" in model_name_or_path or \
            "intfloat" in model_name_or_path or \
            "Snowflake" in model_name_or_path or \
            "GritLM" in model_name_or_path or \
            "Qwen" in model_name_or_path