from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# BLEU
def calculate_bleu(reference, candidate):
    bleu1 = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu([reference], candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4

# ROUGE
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougel = scores['rougeL'].fmeasure
    return rouge1, rouge2, rougel

# Distinct-1
def distinct_1(candidate):
    unigrams = candidate.split()
    return len(set(unigrams)) / len(unigrams)

# Distinct-2
def distinct_2(candidate):
    tokens = candidate.split()
    if len(tokens) < 2:
        return 0
    bigrams = zip(tokens, tokens[1:])
    bigram_list = [" ".join(bigram) for bigram in bigrams]
    return len(set(bigram_list)) / len(bigram_list)

def calculate_metrics(reference, candidate):
    bleu1, bleu2, bleu3, bleu4 = calculate_bleu(reference, candidate)
    rouge1, rouge2, rougel = calculate_rouge(reference, candidate)
    distinct1 = distinct_1(candidate)
    distinct2 = distinct_2(candidate)
    
    return {
        "BLEU-1": bleu1,
        "BLEU-2": bleu2,
        "BLEU-3": bleu3,
        "BLEU-4": bleu4,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougel,
        "Distinct-1": distinct1,
        "Distinct-2": distinct2,
    }
