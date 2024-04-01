import json
import os
from typing import Any, Generator, List, Optional
from llama_index.schema import Document

def save_list_to_json(lst, filename):
    """Save Files

    """
    with open(filename, 'w') as file:
        json.dump(lst, file)

def wr_dict(filename,dic):
    """Write Files

    """
    try:
        if not os.path.isfile(filename):
            data = []
            data.append(dic)
            with open(filename, 'w') as f:
                json.dump(data, f)
        else:      
            with open(filename, 'r') as f:
                data = json.load(f)
                data.append(dic)
            with open(filename, 'w') as f:
                json.dump(data, f)
    except Exception as e:
        print("Save Error:", str(e))
        return
            
def rm_file(file_path):
    """Delete Files

    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} removed successfully.")

def _depth_first_yield(
    json_data: Any,
    levels_back: int,
    collapse_length: Optional[int],
    path: List[str],
    ensure_ascii: bool = False,
) -> Generator[str, None, None]:
    """Do depth first yield of all of the leaf nodes of a JSON.

    Combines keys in the JSON tree using spaces.

    If levels_back is set to 0, prints all levels.
    If collapse_length is not None and the json_data is <= that number
      of characters, then we collapse it into one line.

    """
    if isinstance(json_data, (dict, list)):
        # only try to collapse if we're not at a leaf node
        json_str = json.dumps(json_data, ensure_ascii=ensure_ascii)
        if collapse_length is not None and len(json_str) <= collapse_length:
            new_path = path[-levels_back:]
            new_path.append(json_str)
            yield " ".join(new_path)
            return
        elif isinstance(json_data, dict):
            for key, value in json_data.items():
                new_path = path[:]
                new_path.append(key)
                yield from _depth_first_yield(
                    value, levels_back, collapse_length, new_path
                )
        elif isinstance(json_data, list):
            for _, value in enumerate(json_data):
                yield from _depth_first_yield(value, levels_back, collapse_length, path)
    else:
        new_path = path[-levels_back:]
        new_path.append(str(json_data))
        yield " ".join(new_path)


class JSONReader():
    """JSON reader.

    Reads JSON documents with options to help suss out relationships between nodes.

    """

    def __init__(
        self,
        is_jsonl: Optional[bool] = False,
    ) -> None:
        """Initialize with arguments."""
        super().__init__()
        self.is_jsonl = is_jsonl

    def load_data(self, input_file: str) -> List[Document]:
        """Load data from the input file."""
        
        documents = []
        with open(input_file, 'r') as file:
            load_data = json.load(file)
        for data in load_data:
            metadata = {"title": data['title'], "published_at": data['published_at'],"source":data['source']}
            documents.append(Document(text=data['body'], metadata=metadata))
        return documents
    