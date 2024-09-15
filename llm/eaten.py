import llm.frame_diff.frames as fd

if __name__ == "__main__":
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    fd.generate_frame_diff(model_name)
    fd.determine_eaten()
