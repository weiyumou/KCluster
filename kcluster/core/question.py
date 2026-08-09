from collections import UserDict

from kcluster.core import prompts


class Question(UserDict):
    SPACE = prompts.SPACE

    @property
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

    @property
    def choices(self) -> list[dict]:
        return self["question"].get("choices") or []

    @property
    def body(self) -> str:
        # Rendering follows the data, not the type string: any question that
        # carries choices shows them, so choice-bearing types beyond plain
        # "Multiple Choice" (e.g. select-all) cannot silently lose them.
        if self.choices:  # body = stem + choices
            choices = [prompts.CHOICE_LINE.format(label=item["label"], text=item["text"])
                       for item in self.choices]
            return "\n".join([self.stem] + choices)
        return self.stem  # body = stem

    @property
    def answer(self) -> str:
        return self.get("answerKey", "")

    @property
    def trailer(self) -> str:
        return prompts.ANSWER_TRAILER

    def header(self, q_num: int = 1) -> str:
        hdr = prompts.EXERCISE_HEADER.format(q_num=q_num)
        return f"{hdr}\n{prompts.QUESTION_TYPE_LINE.format(q_type=self.q_type)}" if self.q_type else hdr

    def prompt(self, q_num: int = 1) -> str:
        return f"{self.header(q_num)}\n{self.body}\n{self.trailer}"

    def __str__(self) -> str:
        if self.answer:
            return f"{self.body}\n{self.trailer}{self.SPACE}{self.answer}"
        return self.body
