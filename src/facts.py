"""“Did you know?” facts for the learner app.

Every fact is a short rewording of one sentence in Discover Canada and carries the exact quote it
comes from. tests/test_facts.py checks each quote against the indexed guide text (same page), so a
fact can never drift away from the source.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    text: str          # shown to learners
    page: str          # page of the guide (first page if the passage spans two)
    section: str
    quote: str         # verbatim sentence fragment from the guide (case/punctuation-insensitive match)


FACTS: list[Fact] = [
    Fact("basketball was invented by a Canadian, James Naismith, in 1891?", "26", "Arts and culture in Canada",
         "Basketball was invented by Canadian James Naismith in 1891"),
    Fact("Canadians Sir Frederick Banting and Charles Best discovered insulin, which has saved 16 million lives worldwide?",
         "27", "GREAT CANADIAN DISCOVERIES AND INVENTIONS",
         "discovered insulin, a hormone to treat diabetes that has saved 16 million lives worldwide"),
    Fact("the snowmobile was invented by a Canadian, Joseph-Armand Bombardier?", "27",
         "GREAT CANADIAN DISCOVERIES AND INVENTIONS", "invented the snowmobile"),
    Fact("Alexander Graham Bell hit on the idea of the telephone at his summer house in Canada?", "27",
         "GREAT CANADIAN DISCOVERIES AND INVENTIONS", "hit on the idea of the telephone at his summer house in Canada"),
    Fact("the Canadarm, a robotic arm used in outer space, was invented by SPAR Aerospace and the National Research Council?",
         "27", "GREAT CANADIAN DISCOVERIES AND INVENTIONS", "invented the Canadarm, a robotic arm used in outer space"),
    Fact("in 1980 Terry Fox began his cross-country “Marathon of Hope” to raise money for cancer research?", "25-26",
         "Arts and culture in Canada", "began a cross-country run, the “Marathon of Hope,” to raise money for cancer research"),
    Fact("the Mackenzie River is the second-longest river system in North America, after the Mississippi?", "50",
         "Northwest Territories", "is the second-longest river system in North America after the Mississippi"),
    Fact("Nunavut means “our land” in Inuktitut and was established in 1999?", "51", "Nunavut",
         "Nunavut, meaning “our land” in Inuktitut, was established in 1999"),
    Fact("April 9 is celebrated as Vimy Day, remembering the Canadian victory at Vimy Ridge?", "21",
         "The First World War", "April 9 is celebrated as Vimy Day"),
    Fact("Agnes Macphail, a farmer and teacher, became Canada’s first woman MP in 1921?", "21", "Women get the vote",
         "Agnes Macphail, a farmer and teacher, became the first woman MP in 1921"),
    Fact("ice hockey was developed in Canada in the 1800s?", "39", "Popular sports",
         "Ice hockey was developed in Canada in the 1800s"),
    Fact("on D-Day, June 6, 1944, 15,000 Canadian troops stormed and captured Juno Beach?", "23",
         "The D-Day Invasion, June 6, 1944", "15,000 Canadian troops stormed and captured Juno Beach"),
    Fact("the beaver was adopted centuries ago as a symbol of the Hudson’s Bay Company?", "39", "The beaver",
         "The beaver was adopted centuries ago as a symbol of the Hudson’s Bay Company"),
    Fact("Prince Edward Island is linked to the mainland by the Confederation Bridge, one of the longest continuous "
         "multispan bridges in the world?", "46", "Prince Edward Island",
         "one of the longest continuous multispan bridges in the world"),
]


def pick(rng: random.Random, avoid: int | None = None) -> int:
    """Index of a random fact, different from `avoid` when possible."""
    choices = [i for i in range(len(FACTS)) if i != avoid]
    return rng.choice(choices)
