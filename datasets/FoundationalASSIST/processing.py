"""Driver for the Foundational ASSIST dataset (ASSISTments; gated HuggingFace release).

Cleans the raw ``Problems.csv`` / ``Skills.csv`` / ``Interactions.csv`` export into
two analysis-ready tables plus the Question JSONL the rest of KCluster consumes.
The cleaning rules were worked out in an exploratory notebook that is not
distributed — its saved cell outputs embed problem text and interaction rows from
a gated dataset — so this module is the record: each numbered step carries the
finding that motivated it, and the counts it pins are the evidence.

All three artifacts are written in one pass on purpose. Generating the questions
separately lets ``questions.jsonl`` describe a different problem set than
``problems.csv`` with nothing to signal the drift — every ``--drop_*`` flag
changes both.

The export's own ``Code/clean_utils.py`` does the heavy lifting for MathML, Wiris
formulas, tables and HTML entities. It is loaded from the raw data directory (it
ships with the data, not with this repo) and patched in two places — see
``patch_mathml_parser`` and ``make_clean_body``.

The dataset is gated, licensed CC-BY-NC-4.0, and carries a data-security
undertaking, so the processed tables are written back beside the raw export,
outside this repository. Never commit them.
"""

import argparse
import importlib.util
import os
import re
from fractions import Fraction

import pandas as pd
from bs4 import BeautifulSoup
from unanswerable import REVIEWED_KEY_FIXES, UNANSWERABLE_PROBLEM_IDS

from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, validate_question

# Spanish-language problems. The dataset is otherwise English; these carry a
# separate translation of the same items and would pollute a text-similarity
# space. Four of them ("Encuentra las coordenadas de punto ...") also depend on
# a figure that is not in the text.
SPANISH_PROBLEM_IDS = [307185, 319516, 319607, 319717, 319877, 320133, 320230, 320333, 320443,
                       320538, 322962, 323658, 324288, 324631, 324709, 324806, 324900]

# Individually inspected problems dropped for reasons that are not rule-shaped:
# a non-English item with a nested answer structure, an irrelevant survey
# question, and a "complete the table" item whose three blanks have one answer.
MANUAL_DROP_IDS = [199131, 323395, 343125]

# Pinned count for the truncated-repeating-decimal repair (step 11); a different
# export will have a different number.
EXPECTED_TRUNCATED_KEYS = 7

TEXT_COLS = ("Problem Body", "Fill-in Options", "Fill-in Answers",
             "Multiple Choice Options", "Multiple Choice Answers")

FILL_IN = "Fill-in-the-blank(s)"
SELECT_ONE = "Multiple Choice (select 1)"
SELECT_ALL = "Multiple Choice (select all)"


def load_clean_utils(raw_data_dir: str):
    """Import ``Code/clean_utils.py`` from the raw export directory."""
    path = os.path.join(raw_data_dir, "Code", "clean_utils.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"clean_utils.py not found at {path}; is --raw_data_dir correct?")
    spec = importlib.util.spec_from_file_location("fa_clean_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_mathml_parser(clean_utils) -> None:
    """Teach ``clean_utils.parse_mathml_element`` the container elements it drops.

    The original handles 12 tags and sends everything else to ``get_text()``,
    which flattens nested structure — so a ``<mfrac>`` inside an ``<mfenced>``
    becomes ``14`` instead of ``(1/4)``. We handle the containers explicitly and
    make the fallback recurse, so nesting always survives. Whitespace-only
    ``<mo>`` (e.g. ``<mo>&#160;</mo>``) is rendered as a space — the original
    strips it to nothing, gluing tokens together ("1 to 12" -> "1to12").

    ``clean_problem_body`` resolves ``parse_mathml_element`` from the module
    namespace at call time, so rebinding it here is enough.
    """
    original = clean_utils.parse_mathml_element
    if getattr(original, "_patched", False):
        return

    def kids(elem):
        return [c for c in elem.children if getattr(c, "name", None)]

    def parse(elem):
        if elem.name is None:
            return str(elem).strip()

        if elem.name == "mfenced":  # (a, b) — parentheses were being dropped
            opening, closing = elem.get("open", "("), elem.get("close", ")")
            separators = elem.get("separators", ",")
            parts = [parse(c) for c in kids(elem)]
            if len(parts) <= 1:
                body = "".join(parts)
                # child already supplied parens (mfrac -> "(1/2)"): don't double up
                if opening == "(" and closing == ")" and body.startswith("(") and body.endswith(")"):
                    return body
            else:
                out = [parts[0]]
                for idx, part in enumerate(parts[1:]):
                    sep = separators[idx] if idx < len(separators) else (separators[-1:] or ",")
                    out.append(f"{sep}{part}" if sep.strip() else part)
                body = "".join(out)
            return f"{opening}{body}{closing}"

        if elem.name == "mroot":  # nth root: was rendering ∛27 as "273"
            ch = kids(elem)
            if len(ch) >= 2:
                return f"root{parse(ch[1])}({parse(ch[0])})"

        if elem.name == "mtable":
            return "; ".join(parse(r) for r in kids(elem))
        if elem.name == "mtr":
            return ", ".join(parse(c) for c in kids(elem))
        if elem.name in ("mspace", "msline", "none"):
            return " "

        # Whitespace-only operator (&#160;): the original strips it to "",
        # gluing tokens together ("1 to 12" -> "1to12"). Emit a space, unless
        # it sits inside a number — some bodies use &nbsp; to space out the
        # digits of "0.2", and a space there would split the numeral.
        if elem.name == "mo" and not elem.get_text(strip=True):
            def in_number(sibling):
                if sibling is None:
                    return False
                return sibling.name == "mn" or (sibling.name == "mo" and sibling.get_text(strip=True) == ".")

            if in_number(elem.find_previous_sibling()) and in_number(elem.find_next_sibling()):
                return ""
            return " "

        # tags the original already renders correctly
        if elem.name in ("mfrac", "msup", "msub", "msqrt", "mo", "mn", "mi",
                         "mtext", "mrow", "math", "mpadded", "mstyle"):
            return original(elem)

        # mtd / menclose / mover / munder / mstack / mlongdiv / msrow / msgroup
        # and anything unknown: recurse so nested structure is preserved
        return clean_utils.parse_mathml_element_children(elem)

    parse._patched = True
    clean_utils.parse_mathml_element = parse


def make_clean_body(clean_utils, blank: str = "____"):
    """Build the body cleaner, extending ``clean_problem_body`` with two fixes.

    ``clean_problem_body`` handles MathML, Wiris formulas, tables and HTML
    entities well, but it drops two things that carry meaning in these problems:

    1. ``<ast-r>`` — the ASSISTments answer-blank marker. Fine for a trailing
       blank, but for embedded/multi-blank fill-ins it leaves dangling text
       (e.g. ``1 - = 0.863``). We substitute a visible ``____`` for each,
       numbered when there is more than one so the blanks line up with the
       ``", "``-separated Fill-in Answers.
    2. HTML ``<sup>``/``<sub>`` (distinct from MathML ``<msup>``/``<msub>``),
       which get flattened to a bare number (``10<sup>3</sup>`` -> ``10 3``).
       We render them as ``^``/``_`` (``10^3``, ``cm^2``).
    """

    def clean_body(text):
        if pd.isna(text):
            return ""
        text = str(text)

        # 1. answer blanks
        multi = len(re.findall(r"<ast-r\b[^>]*>", text)) > 1

        def _blank(m):
            k = re.search(r'marker="(\d+)"', m.group(0))
            return f" {blank}({k.group(1)}) " if (multi and k) else f" {blank} "

        text = re.sub(r"<ast-r\b[^>]*>(?:</ast-r>)?", _blank, text)

        # 2. HTML superscripts / subscripts
        soup = BeautifulSoup(text, "html.parser")
        for tag, op in (("sup", "^"), ("sub", "_")):
            for el in soup.find_all(tag):
                inner = el.get_text(strip=True)
                el.replace_with("" if not inner else (f"{op}({inner})" if len(inner) > 1 else f"{op}{inner}"))

        return clean_utils.clean_problem_body(str(soup)).strip()

    return clean_body


DECIMAL = re.compile(r"^-?\d*\.\d+$")


def recover_fraction(key: str, min_places: int = 7, max_denominator: int = 200) -> str | None:
    """The exact fraction a key was rounded from, or None if it was not rounded.

    A handful of answers are stored as the platform's 9-decimal-place rounding
    of a repeating decimal (``-0.055555556`` for ``-1/18``), which no student or
    model writing the exact fraction can ever match. A key qualifies only when
    it has enough places to be machine truncation rather than an exact answer,
    the fraction's expansion genuinely repeats (so ``0.000036 = 9/250`` is left
    alone), and rounding the fraction reproduces the stored digits exactly.
    """
    key = str(key).strip()
    if not DECIMAL.fullmatch(key):
        return None
    places = len(key.split(".")[1])
    if places < min_places:
        return None
    exact = Fraction(key)
    simple = exact.limit_denominator(max_denominator)
    if simple == exact:
        return None
    denominator = simple.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator == 1:  # terminating expansion: the decimal is the exact answer
        return None
    if round(float(simple), places) != float(key):
        return None
    return f"{simple.numerator}/{simple.denominator}"


def contains_pattern(df: pd.DataFrame, pattern: str, cols=TEXT_COLS, **kwargs) -> pd.Series:
    """Row-wise mask: does any of `cols` match `pattern`?"""
    return df[list(cols)].apply(lambda col: col.astype(str).str.contains(pattern, **kwargs)).any(axis=1)


def clean_problems(problems_path: str, clean_body, drop_image_problems: bool = True,
                   drop_unanswerable: bool = True, drop_select_all: bool = False) -> pd.DataFrame:
    """Clean the raw problem table.

    Steps 1-12 are mandatory repairs; 13-15 are filters whose usefulness depends
    on what the table feeds. Each step states the defect it corrects.
    """
    problem_df = pd.read_csv(problems_path)

    # --- Mandatory cleaning ---
    # 1. Drop duplicate problem entries
    problem_df = problem_df.drop_duplicates(
        subset=["Problem Set Id", "Problem Part", "Problem Type", "Answer Types", "Problem Body", "problem_id"],
        ignore_index=True,
    )

    # 2. Fix misclassified MCQs: Fill-in Answers with no Fill-in Options are
    #    multiple-choice problems according to Problem Type.
    mask = problem_df["Fill-in Options"].isna() & problem_df["Fill-in Answers"].notna()
    problem_df.loc[mask, "Fill-in Answers"] = None
    problem_df.loc[mask, "Multiple Choice Answers"] = problem_df.loc[mask, "Multiple Choice Answers"].str.capitalize()

    # 3. Fix answer-options mismatch
    mask = problem_df["Multiple Choice Answers"].notna()
    problem_df.loc[mask, "Multiple Choice Answers"] = problem_df.loc[mask, "Multiple Choice Answers"].str.strip()
    problem_df.loc[problem_df["problem_id"].eq(247727), "Multiple Choice Answers"] = "4.3%"
    problem_df.loc[problem_df["problem_id"].eq(97961), "Multiple Choice Answers"] = "6.7%"
    problem_df.loc[problem_df["problem_id"].eq(496936), ["Fill-in Options", "Fill-in Answers"]] = "60"

    # 4. Re-classify drop-down problems as MCQs
    mask = problem_df["Answer Types"].eq("Drop Down")
    problem_df.loc[mask, "Multiple Choice Options"] = problem_df.loc[mask, "Fill-in Options"].str.replace(",", "||")
    problem_df.loc[mask, "Multiple Choice Answers"] = problem_df.loc[mask, "Fill-in Answers"]
    problem_df.loc[mask, ["Fill-in Options", "Fill-in Answers"]] = None
    problem_df.loc[mask, "Problem Type"] = SELECT_ONE
    problem_df.loc[mask, "Answer Types"] = "Multiple Choice"
    problem_df.loc[problem_df["problem_id"].eq(136977), "Multiple Choice Options"] = "Line segment e || Line segment f"

    # 5. Clean the text columns (clean_body preserves answer blanks and HTML super/subscripts)
    for col in TEXT_COLS:
        problem_df[col] = problem_df[col].apply(clean_body)
    # Some converted dropdown problems have a dangling "____" in the body.
    problem_df.loc[mask, "Problem Body"] = problem_df.loc[mask, "Problem Body"].str.replace("____", "")

    # 6. Fix date-formatted fill-in answers (e.g. "1-Jan" -> "1/1")
    col = problem_df["Fill-in Answers"].str.extract(r"(?i)^(\d+-[a-z]{3}|[a-z]{3}-\d+)$").dropna().squeeze("columns")
    col_df = col.str.split("-", expand=True).rename(columns={0: "day", 1: "month"})
    mask = col_df["day"].str.isalpha()
    col_df.loc[mask, ["day", "month"]] = col_df.loc[mask, ["month", "day"]].values
    col_df["month"] = pd.to_datetime(col_df["month"], format="%b").dt.month
    dt_col = col_df["month"].astype(str) + "/" + col_df["day"].astype(str)
    problem_df.loc[col.index, "Fill-in Options"] = dt_col
    problem_df.loc[col.index, "Fill-in Answers"] = dt_col

    # 7. Fix one particular case of Excel corruption with #NAME?
    mask = problem_df["problem_id"].eq(18775)
    problem_df.loc[mask, ["Fill-in Options", "Fill-in Answers"]] = "-x+2.5"

    # 8. Reverse the order of the Fill-in answers
    mask = problem_df["problem_id"].isin([330621, 227799, 14761, 226545, 351892])
    for col in ["Fill-in Options", "Fill-in Answers"]:
        problem_df.loc[mask, col] = problem_df.loc[mask, col].apply(
            lambda x: ", ".join(reversed(x.split(", "))))

    # 9. Drop select-1 problems with no unique answer (they read as survey
    #    questions), the individually inspected problems, the Spanish-language
    #    problems, and ordering problems, which have no clear answer key.
    mask = problem_df["Problem Type"].eq(SELECT_ONE)
    mask &= problem_df["Multiple Choice Answers"].str.contains(r"\|\|")
    mask |= problem_df["problem_id"].isin(MANUAL_DROP_IDS + SPANISH_PROBLEM_IDS)
    mask |= problem_df["Answer Types"].eq("Ordering")
    problem_df = problem_df[~mask].reset_index(drop=True)

    # 10. Fix the separator in fill-in-the-blanks problems
    mask = (
        problem_df["Problem Type"].eq(FILL_IN)
        & problem_df["Fill-in Options"].str.contains(", ", regex=False)  # potentially multiple options
        & problem_df["Problem Body"].str.count(r"____\(\d\)").ne(2)      # not multi-blank problems
    )
    for col in ["Fill-in Options", "Fill-in Answers"]:
        problem_df.loc[mask, col] = problem_df.loc[mask, col].str.replace(", ", " || ", regex=False)

    # 11. Recover the exact fraction behind a truncated repeating decimal. The
    #     fraction leads because that is what students typed and were graded
    #     correct on; the stored decimal stays as an accepted alternative.
    recovered = []
    for i in problem_df.index[problem_df["Problem Type"].eq(FILL_IN)]:
        decimal = str(problem_df.at[i, "Fill-in Answers"]).strip()
        fraction = recover_fraction(decimal)
        if fraction is None:
            continue
        for col in ("Fill-in Options", "Fill-in Answers"):
            if str(problem_df.at[i, col]).strip() == decimal:
                problem_df.at[i, col] = f"{fraction} || {decimal}"
        recovered.append(int(problem_df.at[i, "problem_id"]))
    assert len(recovered) == EXPECTED_TRUNCATED_KEYS, (
        f"expected {EXPECTED_TRUNCATED_KEYS} truncated repeating decimals, found {len(recovered)}: "
        f"{recovered}. Update EXPECTED_TRUNCATED_KEYS when processing a different export.")
    print(f"** Recovered fractions for {len(recovered)} truncated decimal keys: {recovered} **")

    # 12. Apply the answer keys corrected during the manual review.
    for problem_id, corrected in REVIEWED_KEY_FIXES.items():
        row = problem_df["problem_id"].eq(problem_id)
        problem_df.loc[row, ["Fill-in Options", "Fill-in Answers"]] = corrected

    # --- Optional cleaning (depends on the use case) ---
    # 13. Drop problems that need a figure: a blank image (no alt text), image
    #     options, or alt text describing the figure. None can be answered from
    #     text alone.
    if drop_image_problems:
        has_blank_image = contains_pattern(problem_df, r"\[image\]")
        has_image_options = contains_pattern(
            problem_df, r"\[image[\]:]", cols=("Multiple Choice Options", "Multiple Choice Answers"), case=False)
        has_alt_image = contains_pattern(problem_df, r"\[Image:")
        problem_df = problem_df[~(has_blank_image | has_image_options | has_alt_image)].reset_index(drop=True)

    # 14. Drop problems that cannot be answered from their own text — the
    #     Gemini answerability screen plus its manual review. Keeping them
    #     poisons any text-derived KC model, since the text does not contain
    #     what the question is asking about.
    if drop_unanswerable:
        unanswerable = problem_df["problem_id"].isin(UNANSWERABLE_PROBLEM_IDS)
        print(f"** Dropping {int(unanswerable.sum())} unanswerable problems **")
        problem_df = problem_df[~unanswerable].reset_index(drop=True)

    # 15. Drop select-all-that-apply problems, leaving fill-in-the-blank and
    #     single-answer multiple choice. Their multi-label answer keys need
    #     set-valued scoring that the rest of the pipeline does not model.
    if drop_select_all:
        select_all = problem_df["Problem Type"].eq(SELECT_ALL)
        print(f"** Dropping {int(select_all.sum())} select-all problems **")
        problem_df = problem_df[~select_all].reset_index(drop=True)

    return problem_df


def load_skills(skills_path: str) -> pd.DataFrame:
    """Collapse each problem's skills into one row.

    A problem can map to multiple skills (1-3 here). Drop duplicate skill names
    (different skill_ids can share a node_name), order by name, and join with
    "~~". node_code follows the same name-sorted order, so code[i] pairs with
    name[i].
    """
    skill_df = pd.read_csv(skills_path).drop_duplicates(ignore_index=True)
    return (
        skill_df.drop_duplicates(["problem_id", "node_name"])
        .sort_values(["problem_id", "node_name"])
        .groupby("problem_id", sort=False)
        .agg(
            skill_code=("node_code", lambda codes: "~~".join(codes)),
            skill_name=("node_name", lambda names: "~~".join(names)),
        )
        .reset_index()
    )


def clean_interactions(interactions_path: str, problem_ids) -> pd.DataFrame:
    """Clean the raw interaction log and restrict it to the surviving problems."""
    interaction_df = pd.read_csv(interactions_path).drop(columns=["Unnamed: 0"])
    interaction_df = interaction_df.drop_duplicates(ignore_index=True)

    # 1. Deal with missing answer_text
    mask = interaction_df["answer_text"].isna() & interaction_df["problem_id"].eq(69785)
    interaction_df.loc[mask, "answer_text"] = "[Empty]"

    # 2. Drop rows with missing answer_text, end_time, or discrete_score
    interaction_df = interaction_df.dropna(subset=["answer_text", "end_time", "discrete_score"])

    # 3. Clean answer_text: strip trailing punctuation and the "A. " option labels
    interaction_df["answer_text"] = interaction_df["answer_text"].str.rstrip(". ")
    label = re.compile(r"(?:(?<=^)|(?<=,\s)|(?<=,))[A-Za-z][.)]\s+(?=[^,\s])")
    interaction_df["answer_text"] = interaction_df["answer_text"].str.replace(label, "", regex=True).str.strip()
    interaction_df = interaction_df[~interaction_df["answer_text"].eq("")].reset_index(drop=True)

    # 4. Swap problem_ids between two pairs of problems whose logs were crossed
    swap = {76811: 77198, 77198: 76811, 370522: 370835, 370835: 370522}
    interaction_df["problem_id"] = interaction_df["problem_id"].replace(swap)

    # 5. Drop interactions for problems that were dropped from the problem table
    mask = interaction_df["problem_id"].isin(problem_ids)
    return interaction_df[mask].reset_index(drop=True)


def split_options(text: str) -> list[str]:
    """Split a ``||``-separated option list, dropping empties."""
    return [part.strip() for part in str(text).split("||") if part.strip()]


def build_question(row: pd.Series) -> Question:
    """Convert one cleaned problem row into a Question.

    ==========================  ==============================================
    Question field              Source
    ==========================  ==============================================
    ``id``                      ``fa-<problem_id>``
    ``type``                    ``Problem Type`` verbatim, so both multiple-
                                choice families keep the ``Multiple Choice …``
                                prefix that ``validate_question`` guards on
    ``question.stem``           ``Problem Body``
    ``question.choices``        ``Multiple Choice Options`` split on ``||``
    ``answerKey``               choice label(s) for MCQs, the answer text for
                                fill-ins
    ``skill``/``skill_code``    the expert KC model (CCSS), ``~~`` split
    ==========================  ==============================================

    Select-all answers are the comma-joined labels of every correct option
    (``"b, c"``), matching how ``Question.__str__`` renders a key. Fill-ins have
    no choices, so ``answerKey`` holds the literal answer string — ``" || "``
    still separates alternative acceptable answers and ``", "`` the blanks of a
    multi-blank problem, exactly as the source encodes them.
    """
    q_dict = {
        "id": f"fa-{row['problem_id']}",
        "type": row["Problem Type"],
        "question": {"stem": row["Problem Body"]},
    }

    if row["Problem Type"] == FILL_IN:
        q_dict["answerKey"] = str(row["Fill-in Answers"]).strip()
    else:
        options = split_options(row["Multiple Choice Options"])
        labels = {text: chr(ord("a") + idx) for idx, text in enumerate(options)}
        q_dict["question"]["choices"] = [{"label": labels[text], "text": text} for text in options]

        keys = split_options(row["Multiple Choice Answers"])
        missing = [k for k in keys if k not in labels]
        if missing:
            raise ValueError(f"problem {row['problem_id']}: answer(s) {missing} are not among the options")
        if row["Problem Type"] != SELECT_ALL and len(keys) != 1:
            raise ValueError(f"problem {row['problem_id']}: select-1 problem has {len(keys)} answers")
        q_dict["answerKey"] = ", ".join(sorted(labels[k] for k in keys))

    # Passthrough fields: the expert KC model plus the columns that place a
    # problem in its problem set (used by the answerability screen).
    q_dict["skill"] = str(row["skill_name"]).split("~~")
    q_dict["skill_code"] = str(row["skill_code"]).split("~~")
    q_dict["answer_type"] = row["Answer Types"]
    q_dict["problem_set_id"] = row["Problem Set Id"]
    q_dict["problem_part"] = int(row["Problem Part"])

    return Question(q_dict)


def build_questions(problem_df: pd.DataFrame) -> list[Question]:
    """Convert the cleaned problem table into validated Questions."""
    questions = [build_question(row) for _, row in problem_df.iterrows()]
    for q in questions:
        validate_question(q)
    return questions


def process(raw_data_dir: str, drop_image_problems: bool = True, drop_unanswerable: bool = True,
            drop_select_all: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline, returning the cleaned (problems, interactions)."""
    data_dir = os.path.join(raw_data_dir, "Data")
    clean_utils = load_clean_utils(raw_data_dir)
    patch_mathml_parser(clean_utils)
    clean_body = make_clean_body(clean_utils)

    problem_df = clean_problems(os.path.join(data_dir, "Problems.csv"), clean_body,
                                drop_image_problems, drop_unanswerable, drop_select_all)
    print(f"** Cleaned problems: {len(problem_df)} **")

    # Every problem has >=1 skill, so a left join keeps them all (no fan-out,
    # since skills are pre-aggregated to one row per problem_id).
    problem_df = problem_df.merge(load_skills(os.path.join(data_dir, "Skills.csv")), on="problem_id", how="left")

    interaction_df = clean_interactions(os.path.join(data_dir, "Interactions.csv"), problem_df["problem_id"])
    print(f"** Cleaned interactions: {len(interaction_df)} **")

    # Drop problems left with no interactions
    problem_df = problem_df[problem_df["problem_id"].isin(interaction_df["problem_id"])].reset_index(drop=True)
    print(f"** Problems with >=1 interaction: {len(problem_df)} **")

    return problem_df, interaction_df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_data_dir", default="raw_data", type=str,
                        help="Raw export root, containing Code/ and Data/ (default: raw_data)")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Where to write the processed tables (default: <raw_data_dir>/processed)")
    parser.add_argument("--keep_image_problems", action="store_true",
                        help="Keep problems that depend on a figure (dropped by default)")
    parser.add_argument("--keep_unanswerable", action="store_true",
                        help="Keep problems the answerability screen rejected (dropped by default)")
    parser.add_argument("--drop_select_all", action="store_true",
                        help="Drop select-all-that-apply problems, leaving fill-in and single-answer MCQs")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.raw_data_dir, "processed")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    problem_df, interaction_df = process(args.raw_data_dir,
                                         drop_image_problems=not args.keep_image_problems,
                                         drop_unanswerable=not args.keep_unanswerable,
                                         drop_select_all=args.drop_select_all)

    print("\n** Problem counts by type **")
    print(problem_df.groupby("Problem Type")["Answer Types"].value_counts().to_string())

    for name, df in (("problems.csv", problem_df), ("interactions.csv", interaction_df)):
        path = os.path.join(output_dir, name)
        df.to_csv(path, index=False)
        print(f"** Saved {len(df)} rows to {path} **")

    # Written in the same pass as the tables: a questions.jsonl produced
    # separately can silently describe a different problem set than the CSV.
    questions_path = os.path.join(output_dir, "questions.jsonl")
    questions = build_questions(problem_df)
    dump_questions(questions, questions_path)
    print(f"** Saved {len(questions)} questions to {questions_path} **")


if __name__ == "__main__":
    main()
