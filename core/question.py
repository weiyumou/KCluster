from collections import UserDict
from functools import cached_property


class Question(UserDict):
    SPACE = chr(32)

    @cached_property
    def flat_dict(self) -> dict:
        """Returns a flattened dict repr"""
        q_dict = dict(self.data)
        q_dict["question"] = str(self)
        for field in q_dict:
            if isinstance(q_dict[field], (list, tuple)) and all(isinstance(item, str) for item in q_dict[field]):
                q_dict[field] = "~".join(q_dict[field])
        return q_dict

    @property
    def q_type(self) -> str:
        return self.get("type", "")

    @property
    def stem(self) -> str:
        return self["question"]["stem"]

    @cached_property
    def body(self) -> str:
        if self.q_type == "Multiple Choice":  # body = stem + choices
            choices = [f"{item['label']}){self.SPACE}{item['text']}" for item in self["question"]["choices"]]
            return "\n".join([self.stem] + choices)
        return self.stem  # body = stem

    @property
    def answer(self) -> str:
        return self.get("answerKey", "")

    @property
    def trailer(self) -> str:
        return "Answer:"

    def header(self, q_num: int = 1) -> str:
        hdr = f"Exercise{self.SPACE}{q_num}:"
        return f"{hdr}\n{self.q_type}:" if self.q_type else hdr

    def prompt(self, q_num: int = 1) -> str:
        return f"{self.header(q_num)}\n{self.body}\n{self.trailer}"

    def __str__(self) -> str:
        return f"{self.body}\n{self.trailer}{self.SPACE}{self.answer}" if self.answer else self.body
