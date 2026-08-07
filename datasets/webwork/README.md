# WeBWorK (Rogawski Calculus)

Driver for the WeBWorK answer log covering Rogawski Calculus problems.
`processing.py` cleans the raw student data (problem-path corrections,
student/subject filters, chapter/section backfills), extracts problem
content from the raw problem files, and reshapes everything into
DataShop-style transactions (one row per answer blank); it also derives
Chapter/Section and Keywords KC models from the transaction data.

Run from this directory, e.g.:

    python processing.py --raw_stu_data_path raw_data/<log>.csv --data_dir data/

The `expected_records` check (27071) pins the record count of the original
export; pass a different value when processing another export.
