import json
import os
from argparse import ArgumentParser
from typing import Dict, List
from pprint import pprint

START_IMP_SEQUENCE = "# START PROJECT IMPORTS"
END_IMP_SEQUENCE = "# END PROJECT_IMPORTS"
MARKDOWN_SEQUENCE = "<markdown>"
CODE_SEQUENCE = "<code>"


def read_file(path: str) -> List[str]:
    source = []
    add_lines = True
    for code_line in open(path, 'r'):
        if add_lines and code_line.startswith(START_IMP_SEQUENCE):
            add_lines = False
            continue

        if not add_lines and code_line.startswith(END_IMP_SEQUENCE):
            add_lines = True
            continue

        if add_lines:
            source.append(code_line)

    return source


def add_code(notebook: Dict, source: List[str]) -> None:
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        }
    )


def add_markdown(notebook: Dict, source: List[str]) -> None:
    notebook["cells"].append(
        {
            "cell_type": "markdown",
            "source": source
        }
    )


def convert_to_jupyter(config_path, src, verbose):
    if not os.path.exists("convert_order.json"):
        print("Please create convert_order.json in script directory and fill it with needed data")
        exit(1)
    convert_order = json.load(open("convert_order.json", 'r'))

    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    project_files = {}
    # parse all python files in the project
    for subdir, dirs, files in os.walk(src):
        for file in files:
            fullpath = os.path.join(subdir, file)
            project_files[file] = fullpath

    # preprocess and add each cell
    for cell in convert_order["cells"]:
        if cell.startswith(MARKDOWN_SEQUENCE):
            source = cell[len(MARKDOWN_SEQUENCE):].split("\n")
            for idx, line in enumerate(source):
                source[idx] = line + "\n"
            add_markdown(notebook, source)
            continue

        if cell.startswith(CODE_SEQUENCE):
            filename = cell[len(CODE_SEQUENCE):]
            if filename in project_files:
                source = read_file(project_files[filename])
            else:
                source = filename.split("\n")
                for idx, line in enumerate(source):
                    source[idx] = line + "\n"
            add_code(notebook, source)

    # add config and predict call
    config = json.load(open(config_path, 'r'))
    config["submission_read_dir"] = "/kaggle/input/sartorius-cell-instance-segmentation"
    config["submission_csv_dir"] = "/kaggle/input/sartorius-cell-instance-segmentation/sample_submission.csv"
    config_str = json.dumps(config, indent=4)
    source = config_str.split("\n")
    for idx, line in enumerate(source):
        source[idx] = line.replace("null", "None").replace("true", "True").replace("false", "False") + "\n"
    add_markdown(notebook, ["Create configuration dict\n"])
    add_code(notebook, ["cnf=\\\n", *source])

    add_code(notebook, ["predict_submission(cnf, \"val_best.h5\")\n"])

    if verbose:
        pprint(notebook)

    json.dump(notebook, open("submission.ipynb", 'w'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--src", type=str, default="src")
    parser.add_argument("--verbose", action="store_const", const=True, default=False)
    args = parser.parse_args()
    convert_to_jupyter(args.config, args.src, args.verbose)
