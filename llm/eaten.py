import llm.frame_diff.frames as fd
import llm.analysis.analyze as an
from rich.console import Console

if __name__ == "__main__":
    """
    # frame_diff

    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    fd.generate_frame_diff(model_name)
    fd.determine_eaten()
    """

    """
    # analysis

    # Test 1 - hardcoded lists
    llm_lst = ["hamburger", "lettuce", "tomato", "pickles", "bun"]
    gt_lst = ["spinach", "pickles || relish", "bread roll"]
    diff = an.match_outputs(llm_lst, gt_lst)
    score = an.compute_diff_score(diff)
    console = Console()
    print()
    console.print(f"Score: {score}", style="yellow")
    print()
    console.print("Diff: ", style="yellow")
    for key, value in diff.items():
        console.print(f"[bold]{key}[/bold]: {value}", style="blue")
    print()

    # Test 2 - video analysis
    diff, score = an.compare_pred(video_name='2.mp4', frame_number=1100, prompt_index=an.INGREDIENTS_PROMPT_INDEX)
    console = Console()
    print()
    console.print(f"Score: {score}", style="yellow")
    print()
    console.print("Diff: ", style="yellow")
    for key, value in diff.items():
        console.print(f"[bold]{key}[/bold]: {value}", style="blue")
    print()
    """
