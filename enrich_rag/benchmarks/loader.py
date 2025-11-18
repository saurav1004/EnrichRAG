import json

def load_benchmark_data(dataset_path, loader_name):
    """General-purpose data loader."""
    if loader_name == "load_nq":
        return load_nq(dataset_path)
    elif loader_name == "load_triviaqa":
        return load_triviaqa(dataset_path)
    elif loader_name == "load_popqa":
        return load_popqa(dataset_path)
    elif loader_name == "load_hotpotqa":
        return load_hotpotqa(dataset_path)
    elif loader_name == "load_2wikimultihopqa":
        return load_2wikimultihopqa(dataset_path)
    elif loader_name == "load_musique":
        return load_musique(dataset_path)
    else:
        raise ValueError(f"Unknown dataset_loader: {loader_name}")

def load_nq(path):
    """Loads Natural Questions data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            data.append({
                "id": item.get("id", str(len(data))),
                "question": item["question"],
                "answers": item["golden_answers"] # List of strings
            })
    return data

def load_triviaqa(path):
    """Loads TriviaQA data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            data.append({
                "id": item.get("id", str(len(data))),
                "question": item["question"],
                "answers": item["golden_answers"] # List of strings
            })
    return data

def load_popqa(path):
    """Loads PopQA data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            data.append({
                "id": item.get("id", str(len(data))),
                "question": item["question"],
                "answers": item["golden_answers"] # List of strings
            })
    return data

def load_hotpotqa(path):
    """Loads HotpotQA data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            data.append({
                "id": item.get("id", str(len(data))),
                "question": item["question"],
                "answers": item["golden_answers"] # List of strings
            })
    return data

def load_2wikimultihopqa(path):
    """Loads 2WikiMultiHopQA data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            data.append({
                "id": item.get("id", str(len(data))),
                "question": item["question"],
                "answers": item["golden_answers"] # List of strings
            })
    return data

def load_musique(path):
    """Loads Musique data."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Standardize format
            data.append({
                "id": item.get("id", str(len(data))),
                "question": item["question"],
                "answers": item["golden_answers"] # List of strings
            })
    return data