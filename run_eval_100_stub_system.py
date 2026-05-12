import importlib.machinery
import contextlib
import os
import runpy
import sys
import types

ROOT = r"F:\major project V1\major project"
CACHE = os.path.join(ROOT, "hf_cache_run4")
os.makedirs(os.path.join(CACHE, "tmp"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "local_pkgs_run4"), exist_ok=True)

os.environ["HF_HOME"] = CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE, "datasets")
os.environ["TEMP"] = os.path.join(CACHE, "tmp")
os.environ["TMP"] = os.path.join(CACHE, "tmp")
os.environ["PIP_CACHE_DIR"] = os.path.join(CACHE, "pip-cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

TRACE_PATH = os.path.join(ROOT, "run_eval_100_stub_system.trace.log")

SAMPLES = [
    (
        "What is the capital of France?",
        "Paris",
        "France_Capital",
        "Paris is the capital and largest city of France.",
        "Mars",
        "Mars is the fourth planet from the Sun.",
    ),
    (
        "Which planet is known as the Red Planet?",
        "Mars",
        "Red_Planet",
        "Mars is often called the Red Planet because of iron oxide on its surface.",
        "Venus",
        "Venus is the second planet from the Sun.",
    ),
    (
        "What gas do plants absorb from the air?",
        "Carbon dioxide",
        "Photosynthesis",
        "Plants absorb carbon dioxide from the air during photosynthesis.",
        "Oxygen",
        "Humans and many animals breathe oxygen.",
    ),
    (
        "What is the tallest mountain on Earth?",
        "Mount Everest",
        "Everest",
        "Mount Everest is the tallest mountain on Earth above sea level.",
        "K2",
        "K2 is the second highest mountain on Earth.",
    ),
    (
        "Which ocean is the largest on Earth?",
        "Pacific Ocean",
        "Pacific",
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        "Atlantic Ocean",
        "The Atlantic Ocean is the second-largest ocean on Earth.",
    ),
    (
        "Who wrote Hamlet?",
        "William Shakespeare",
        "Hamlet_Author",
        "William Shakespeare wrote the tragedy Hamlet.",
        "Charles Dickens",
        "Charles Dickens wrote novels such as Great Expectations.",
    ),
    (
        "What is the chemical symbol for water?",
        "H2O",
        "Water_Symbol",
        "The chemical formula for water is H2O.",
        "CO2",
        "Carbon dioxide is commonly written as CO2.",
    ),
    (
        "What device do people use to call someone?",
        "phone",
        "Telephone",
        "A phone is used to call and speak with other people over a network.",
        "Book",
        "A book contains written or printed pages.",
    ),
    (
        "What is the fastest land animal?",
        "cheetah",
        "Cheetah",
        "The cheetah is the fastest land animal.",
        "Elephant",
        "Elephants are the largest land animals.",
    ),
    (
        "Which instrument has keys, pedals, and strings?",
        "piano",
        "Piano",
        "A piano has keys, pedals, and strings inside the instrument.",
        "Drum",
        "A drum is a percussion instrument.",
    ),
]


def build_samples(n: int = 100):
    items = []
    for i in range(n):
        question, answer, support_title, support_text, distractor_title, distractor_text = SAMPLES[i % len(SAMPLES)]
        items.append(
            {
                "id": f"fake-{i}",
                "question": question,
                "answer": answer,
                "supporting_facts": {"title": [support_title]},
                "context": {
                    "title": [support_title, distractor_title],
                    "sentences": [[support_text], [distractor_text]],
                },
            }
        )
    return items


def load_dataset(name, config=None, split=None):
    if name != "hotpot_qa":
        raise ValueError(f"Unsupported dataset: {name}")
    return build_samples(100)


datasets_module = types.ModuleType("datasets")
datasets_module.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
datasets_module.load_dataset = load_dataset
sys.modules["datasets"] = datasets_module

sys.argv = ["main.py", "--mode", "evaluate", "--samples", "100"]

with open(TRACE_PATH, "w", encoding="utf-8") as trace_file:
    with contextlib.redirect_stdout(trace_file), contextlib.redirect_stderr(trace_file):
        print("=== synthetic 100-sample run started ===")
        print(f"HF_HOME={os.environ['HF_HOME']}")
        print("Running main.py --mode evaluate --samples 100")
        try:
            runpy.run_path("main.py", run_name="__main__")
            print("=== synthetic 100-sample run finished ===")
        except Exception as exc:
            print(f"=== synthetic 100-sample run failed: {exc!r} ===")
            raise
