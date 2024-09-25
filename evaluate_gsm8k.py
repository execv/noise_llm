import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.generation import GenerationConfig


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

class GaussianNoiseLogitsProcessor(LogitsProcessor):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, input_ids, scores):
        # 添加高斯噪声到 logits (即 scores)
        noise = torch.normal(mean=self.mean, std=self.std, size=scores.shape).to(scores.device)
        return scores + noise

def doc_to_text(doc):
    return (
        fewshot_prompt
        + "\nQuestion: "
        + doc["question"]
        + "\nLet's think step by step\n"
    )


def decode(tokens_list, tokenizer, raw_text_len):
    sents = []
    # print(len(tokens_list))
    for tokens in tokens_list:
        tokens = tokens.cpu().numpy().tolist()
        sent = tokenizer.tokenizer.decode(tokens[raw_text_len:])
        sent = sent.split("<|endoftext|>")[0]
        sent = sent.split("\n\n\n")[0]
        sent = sent.split("\n\n")[0]
        sent = sent.split("Question:")[0]
        sents.append(sent)
    return sents


def generate_sample(model, tokenizer, input_txt, logits_processor_list):
    input_ids = tokenizer.tokenizer.encode(input_txt)
    raw_text_len = len(input_ids)
    context_enc = torch.tensor([input_ids]).to(model.device)
    outputs = model.generate(context_enc, logits_processor=logits_processor_list)
    output_text = decode(outputs, tokenizer, raw_text_len)[0]
    return output_text


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="Qwen/Qwen-7B",
    )
    parser.add_argument("-f", "--sample-input-file", type=str, default=None)
    parser.add_argument(
        "-o", "--sample-output-file", type=str, default="gsm8k_res.jsonl"
    )

    args = parser.parse_args()

    fewshot_prompt = open("gsm8k_prompt.txt").read()
    if args.sample_input_file is not None:
        dataset = load_from_disk(args.sample_input_file)
    else:
        config = datasets.DownloadConfig(resume_download=True, max_retries=100)
        dataset = load_dataset("gsm8k", "main", download_config=config)

    test = dataset["test"]

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )

    print("Loading model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path, device_map="auto", trust_remote_code=True
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True
    )
    model.generation_config.do_sample = False

    for x in range(0, 50):
        std = x / 10
        f_output = jsonlines.Writer(open(f"{args.sample_output_file}_std_{std}", "w", encoding="utf-8"))
        tot_length = test.num_rows
        logits_processor = LogitsProcessorList()
        logits_processor.append(GaussianNoiseLogitsProcessor(mean=0.0, std=std))
        acc_res = []
        for doc in test:
            context = doc_to_text(doc)
            completion = generate_sample(model, tokenizer, context, logits_processor_list=logits_processor)
            answer = doc["answer"]
            acc = is_correct(completion, answer)
            doc["completion"] = completion
            doc["acc"] = acc
            f_output.write(doc)
            acc_res.append(acc)

        f_output.close()
        print(f"Std: {std}\tAcc: {np.mean(acc_res)}")
