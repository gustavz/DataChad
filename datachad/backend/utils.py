import re


def clean_string_for_storing(string: str) -> str:
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    cleaned_string = re.sub(r"\W+", "-", string)
    cleaned_string = re.sub(r"--+", "- ", cleaned_string).strip("-")
    return cleaned_string
