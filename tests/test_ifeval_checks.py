"""IFEval-style constraint verifiers + the fraction-satisfied grader."""

from studio.benchmark.ifeval_checks import check, supported
from studio.benchmark.qa import QATask
from studio.benchmark.qa_suites import _grade_ifeval


def test_supported_and_unknown():
    assert supported("punctuation:no_comma")
    assert not supported("language:response_language")  # intentionally unimplemented
    assert check("language:response_language", "x", {}) is False  # unknown -> False


def test_no_comma():
    assert check("punctuation:no_comma", "no commas here", {}) is True
    assert check("punctuation:no_comma", "yes, there is", {}) is False


def test_keyword_existence_and_forbidden():
    assert check("keywords:existence", "the Eiffel tower", {"keywords": ["eiffel"]}) is True
    assert check("keywords:existence", "nothing", {"keywords": ["eiffel"]}) is False
    assert check("keywords:forbidden_words", "clean text", {"forbidden_words": ["bad"]}) is True
    assert check("keywords:forbidden_words", "this is bad", {"forbidden_words": ["bad"]}) is False


def test_number_words_relation():
    text = "one two three four five"
    assert check("length_constraints:number_words", text, {"num_words": 5, "relation": "at least"}) is True
    assert check("length_constraints:number_words", text, {"num_words": 6, "relation": "at least"}) is False
    assert check("length_constraints:number_words", text, {"num_words": 6, "relation": "less than"}) is True


def test_case_and_endcheck_and_title():
    assert check("change_case:english_lowercase", "all lower", {}) is True
    assert check("change_case:english_lowercase", "Has Caps", {}) is False
    assert check("startend:end_checker", "wrap it up. that's all", {"end_phrase": "that's all"}) is True
    assert check("detectable_format:title", "<<My Title>>\nbody", {}) is True
    assert check("detectable_format:json_format", '```json\n{"a":1}\n```', {}) is True
    assert check("detectable_format:json_format", "not json", {}) is False


def test_grade_ifeval_fraction_satisfied():
    t = QATask(id="0", question="q", meta={
        "instruction_id_list": ["punctuation:no_comma", "keywords:existence"],
        "kwargs": [{}, {"keywords": ["cat"]}],
    })
    assert _grade_ifeval("a cat with no commas", t) == 1.0     # both satisfied
    assert _grade_ifeval("a cat, with comma", t) == 0.5        # only keyword ok
    assert _grade_ifeval("dog, here", t) == 0.0                # neither
