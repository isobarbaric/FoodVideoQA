from pathlib import Path
import numpy as np
from bert_score import BERTScorer, plot_example
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from rich.console import Console

console = Console()
scorer = BERTScorer(
    model_type="bert-large-uncased", lang="en", rescale_with_baseline=True
)


def preprocess(str_lst: list[str]) -> list[str]:
    # print(' '.join(str_lst))
    return [" ".join(str_lst)]


def semantic_matching(
    llm_lst: list[str], gt_lst: list[str], plot=False
) -> tuple[float, float, float]:
    llm_lst_str = preprocess(llm_lst)
    gt_lst_str = preprocess(gt_lst)

    print(llm_lst_str[0])
    print(gt_lst_str[0])

    if llm_lst_str[0].strip() == "" or gt_lst_str[0].strip() == "":
        return 1, 1, 1

    P, R, F1 = scorer.score(llm_lst_str, gt_lst_str)

    if plot:
        plot_example(
            llm_lst_str[0],
            gt_lst_str[0],
            model_type="bert-large-uncased",
            lang="en",
            rescale_with_baseline=True,
            fname="llm/analysis/matrix.png",
        )

    return float(P), float(R), float(F1)


def word_matching(llm_lst: list[str], gt_lst: list[str]):
    llm_lst_c = np.array(llm_lst.copy())
    gt_lst_c = np.array(gt_lst.copy())

    if len(llm_lst_c) > len(gt_lst_c):
        gt_lst_c = np.pad(
            gt_lst_c,
            (0, len(llm_lst_c) - len(gt_lst_c)),
            "constant",
            constant_values="",
        )
    elif len(gt_lst_c) > len(llm_lst_c):
        llm_lst_c = np.pad(
            llm_lst_c,
            (0, len(gt_lst_c) - len(llm_lst_c)),
            "constant",
            constant_values="",
        )

    P, R, F1, _ = precision_recall_fscore_support(llm_lst_c, gt_lst_c, average="micro")
    return P, R, F1


# def perform_matching(llm_lst, gt_lst):
#     P1_w , R1_w, F1_w = word_matching(llm_lst, gt_lst)
#     P1_s, R1_s, F1_s = semantic_matching(llm_lst, gt_lst)
#     results = {
#         'word-matching': {
#             'P': P1_w,
#             'R': R1_w,
#             'F1': F1_w
#         }
#         'semantic-matching': {
#             'P': P1_s,
#             'R': R1_s,
#             'F1': F1_s
#         }
#     }
#     return results


if __name__ == "__main__":
    f1_word_matching = []
    f1_semantic_matching = []

    # Test #0
    llm_lst = ["pasta", "meatballs", "tomato sauce", "parmesan"]
    gt_lst = ["spaghetti", "meatballs", "marinara", "cheese"]
    console.print(f"LLM List: {llm_lst}", style="bold cyan")
    console.print(f"Ground Truth List: {gt_lst}", style="bold magenta")
    P, R, F1 = word_matching(llm_lst, gt_lst)
    f1_word_matching.append(F1)
    console.print(f"one-to-one word matching - P: {P}, R: {R}, F1: {F1}", style="green")
    P, R, F1 = semantic_matching(llm_lst, gt_lst, plot=False)
    f1_semantic_matching.append(F1)
    console.print(f"semantic matching: P: {P}, R: {R}, F1: {F1}", style="blue")
    console.print()

    # Test #1
    llm_lst = ["omelette", "egg", "cucumber", "fries", "potato", "tomato"]
    gt_lst = ["carrot", "egg", "tomato", "potato fries"]
    console.print(f"LLM List: {llm_lst}", style="bold cyan")
    console.print(f"Ground Truth List: {gt_lst}", style="bold magenta")
    P, R, F1 = word_matching(llm_lst, gt_lst)
    f1_word_matching.append(F1)
    console.print(f"one-to-one word matching - P: {P}, R: {R}, F1: {F1}", style="green")
    P, R, F1 = semantic_matching(llm_lst, gt_lst, plot=True)
    f1_semantic_matching.append(F1)
    console.print(f"semantic matching: P: {P}, R: {R}, F1: {F1}", style="blue")
    console.print()

    # Test #2
    llm_lst = ["spaghetti", "pasta", "meatballs", "tomato sauce", "cheese"]
    gt_lst = ["spaghetti", "meatball", "tomato sauce", "cheese"]
    console.print(f"LLM List: {llm_lst}", style="bold cyan")
    console.print(f"Ground Truth List: {gt_lst}", style="bold magenta")
    P, R, F1 = word_matching(llm_lst, gt_lst)
    f1_word_matching.append(F1)
    console.print(f"one-to-one word matching - P: {P}, R: {R}, F1: {F1}", style="green")
    P, R, F1 = semantic_matching(llm_lst, gt_lst, plot=True)
    f1_semantic_matching.append(F1)
    console.print(f"semantic matching: P: {P}, R: {R}, F1: {F1}", style="blue")
    console.print()

    # Test #3
    llm_lst = ["pasta", "alfredo sauce", "chicken", "broccoli", "parmesan"]
    gt_lst = ["pasta", "alfredo sauce", "chicken", "broccoli", "parmesan"]
    console.print(f"LLM List: {llm_lst}", style="bold cyan")
    console.print(f"Ground Truth List: {gt_lst}", style="bold magenta")
    P, R, F1 = word_matching(llm_lst, gt_lst)
    f1_word_matching.append(F1)
    console.print(f"one-to-one word matching - P: {P}, R: {R}, F1: {F1}", style="green")
    P, R, F1 = semantic_matching(llm_lst, gt_lst, plot=True)
    f1_semantic_matching.append(F1)
    console.print(f"semantic matching: P: {P}, R: {R}, F1: {F1}", style="blue")
    console.print()

    # Test #4
    llm_lst = ["hamburger", "lettuce", "tomato", "pickles", "bun"]
    gt_lst = ["spinach", "pickles", "bread roll"]
    console.print(f"LLM List: {llm_lst}", style="bold cyan")
    console.print(f"Ground Truth List: {gt_lst}", style="bold magenta")
    P, R, F1 = word_matching(llm_lst, gt_lst)
    f1_word_matching.append(F1)
    console.print(f"one-to-one word matching - P: {P}, R: {R}, F1: {F1}", style="green")
    P, R, F1 = semantic_matching(llm_lst, gt_lst, plot=True)
    f1_semantic_matching.append(F1)
    console.print(f"semantic matching: P: {P}, R: {R}, F1: {F1}", style="blue")
    console.print()

    avg_f1_word = sum(f1_word_matching) / len(f1_word_matching)
    avg_f1_semantic = sum(f1_semantic_matching) / len(f1_semantic_matching)

    console.print(
        f"Average F1 Score - Word Matching: {avg_f1_word:.2f}", style="bold yellow"
    )
    console.print(
        f"Average F1 Score - Semantic Matching: {avg_f1_semantic:.2f}",
        style="bold yellow",
    )
