"""Regression tests for concept-hub semantic validation, built from the three diagrams that were wrong:
"First Nation Languages", "Iroquoian" and "Aboriginal languages". The passages are the guide's own text (p.10, p.11, p.14); the raw nodes
are what the model produced for them (and the plausible mistakes around them). No model and no API key are needed."""
import pytest

from src import concept_semantics as cs
from src import learning_validation as lv
from src.learning_models import ConceptHub, ConceptNode
from src.retrieval import RetrievedChunk

P11 = ("Who We Are Unity in Diversity The Inuit, which means “the people” in the Inuktitut language, live in small, scattered communities across "
       "the Arctic. Their knowledge of the land, sea and wildlife enabled them to adapt to one of the harshest environments on earth. The Métis are a "
       "distinct people of mixed Aboriginal and European ancestry, the majority of whom live in the Prairie provinces. They come from both French- and "
       "English-speaking backgrounds and speak their own dialect, Michif. About 65% of the Aboriginal people are First Nations, while 30% are Métis and 4% Inuit.")
P14 = ("Canada’s History Exploring a river, naming Canada Between 1534 and 1542, Jacques Cartier made three voyages across the Atlantic, claiming the land for "
       "King Francis I of France. Cartier heard two captured guides speak the Iroquoian word kanata, meaning “village.” By the 1550s, the name of Canada began appearing on maps.")
P10 = ("Who We Are Aboriginal peoples From the 1800s until the 1980s, the federal government placed many Aboriginal children in residential schools to educate "
       "and assimilate them into mainstream Canadian culture. The schools were poorly funded and inflicted hardship on the students; some were physically abused. "
       "Aboriginal languages and cultural practices were mostly prohibited. In 2008, Ottawa formally apologized to the former students.")
SRC = {"S1": RetrievedChunk("S1", "c11", P11, "11", "Unity in Diversity", "p"),
       "S2": RetrievedChunk("S2", "c14", P14, "14", "Exploring a river, naming Canada", "p"),
       "S3": RetrievedChunk("S3", "c10", P10, "10", "Aboriginal peoples", "p")}

Q_INUIT = "The Inuit, which means “the people” in the Inuktitut language, live in small, scattered communities across the Arctic."
Q_METIS = "They come from both French- and English-speaking backgrounds and speak their own dialect, Michif."
Q_KANATA = "Cartier heard two captured guides speak the Iroquoian word kanata, meaning “village.”"
Q_NAME = "By the 1550s, the name of Canada began appearing on maps."
Q_SCHOOLS = "Aboriginal languages and cultural practices were mostly prohibited."


def node(label, etype, predicate, quote, src, subject_is_center=True, relationship=""):
    return {"label": label, "entity_type": etype, "predicate": predicate, "subject_is_center": subject_is_center, "relationship": relationship,
            "explanation": "x", "sources": [src], "evidence_quote": quote}


def run(center, center_type, nodes):
    return lv.validate_hub_nodes(nodes, center, SRC, 8, center_type)


def by_label(nodes):
    return {n.label: n for n in nodes}


# --- 1. "First Nation Languages": Inuit and Métis are not First Nations -----------------------------------------------------------

def test_first_nation_languages_does_not_hold_inuit_or_metis_items():
    nodes, omitted = run("First Nation Languages", "language", [
        node("Inuktitut", "language", "part_of", Q_INUIT, "S1", subject_is_center=False),
        node("Inuit", "people", "speaks", Q_INUIT, "S1", subject_is_center=False),
        node("Michif", "language", "part_of", Q_METIS, "S1", subject_is_center=False),
        node("Métis", "people", "speaks", Q_METIS, "S1", subject_is_center=False),
    ])
    assert nodes == []                                                           # the guide names no First Nations language here
    assert len(omitted) == 4 and all("not the First Nations" in o for o in omitted)
    assert any("Inuit" in o and "places it with the Inuit" in o for o in omitted)
    assert any("Michif" in o and "places it with the Métis" in o for o in omitted)   # found through the previous sentence ("The Métis are…")


def test_the_old_summary_that_mixed_peoples_is_dropped():
    old = ("The sources connect Aboriginal language identities to specific peoples: the Inuit are identified through the Inuktitut language, "
           "and the Métis speak their own dialect, Michif.")
    assert lv.validate_summary(old, ["S1"], SRC, "First Nation Languages") == ("", [])
    assert lv.validate_summary(old, ["S1"], SRC, "Aboriginal languages")[0] == old       # fine for the umbrella term


# --- 2. "Iroquoian": a word is not a language family, and derivation must be stated ---------------------------------------

def test_iroquoian_keeps_the_word_with_the_right_direction_and_does_not_claim_a_derivation():
    nodes, omitted = run("Iroquoian", "language_family", [
        node("kanata", "word", "word_in", Q_KANATA, "S2", subject_is_center=False),
        node("Jacques Cartier", "person", "heard_by", Q_KANATA, "S2", subject_is_center=True, relationship="word heard by"),
        node("the name of Canada", "concept", "name_derived_from", Q_NAME, "S2", subject_is_center=True),
    ])
    n = by_label(nodes)
    assert n["kanata"].relationship == "includes the word" and n["kanata"].support == "stated" and n["kanata"].entity_type == "word"
    # reads "Iroquoian includes the word kanata": the word is the subject, the family the object
    assert n["Jacques Cartier"].relationship == "is mentioned with" and n["Jacques Cartier"].support == "co-mentioned"     # a family is not a word that was heard
    assert n["the name of Canada"].relationship == "is mentioned with"                                                      # derivation is NOT stated in the passage
    assert not any(x.relationship.startswith(("takes its name", "is the source")) for x in nodes)
    assert sum("revised" in o for o in omitted) == 2


def test_a_derivation_is_accepted_only_when_the_quote_says_so():
    said = "The name Canada comes from the Iroquoian word kanata, meaning village."
    ok = cs.check_relationship(center="kanata", center_type="word", label="Canada", node_type="place", predicate="name_derived_from", subject_is_center=False,
                               relationship_text="", quote=said, chunk_text=said)
    assert ok.status == "ok" and ok.support == "stated" and ok.display == "is the source of the name of"
    no = cs.check_relationship(center="kanata", center_type="word", label="Canada", node_type="place", predicate="name_derived_from", subject_is_center=False,
                               relationship_text="", quote=Q_NAME + " " + Q_KANATA, chunk_text=P14)
    assert no.status == "revised" and no.display == "is mentioned with"


# --- 3. "Aboriginal languages": language/people/institution, and what is stated ----------------------------------------------

def test_aboriginal_languages_keeps_the_institution_and_weakens_unstated_links():
    nodes, omitted = run("Aboriginal languages", "language", [
        node("Residential schools", "institution", "prohibited_in", Q_SCHOOLS, "S3", subject_is_center=True),
        node("Inuktitut", "language", "part_of", Q_INUIT, "S1", subject_is_center=False),
        node("Michif", "language", "speaks", Q_METIS, "S1", subject_is_center=True),        # wrong: a language cannot be the subject of "speaks"
    ])
    n = by_label(nodes)
    assert n["Residential schools"].relationship == "prohibited in" and n["Residential schools"].support == "stated"
    assert n["Residential schools"].category == "organization"                                  # an institution, not a generic concept
    assert n["Inuktitut"].relationship == "is mentioned with"                                  # "part of/among/includes" is not stated
    assert n["Michif"].relationship == "is mentioned with"                                     # types do not fit "speaks" with the language as subject
    assert not any(o for o in omitted if "rejected" in o)


# --- 4. direction -------------------------------------------------------------------------------------------------------------

def test_a_reversed_arrow_is_corrected_when_the_types_make_it_unambiguous():
    v = cs.check_relationship(center="Michif", center_type="language", label="Métis", node_type="people", predicate="speaks", subject_is_center=True,
                              relationship_text="", quote=Q_METIS, chunk_text=P11)
    assert v.status == "revised" and v.subject_is_center is False and v.display == "is spoken by" and v.reason == "direction corrected"
    right = cs.check_relationship(center="Métis", center_type="people", label="Michif", node_type="language", predicate="speaks", subject_is_center=True,
                                  relationship_text="", quote=Q_METIS, chunk_text=P11)
    assert right.status == "ok" and right.display == "speak"


@pytest.mark.parametrize("center_type,node_type,predicate,subject_is_center", [
    ("people", "people", "speaks", True),                 # a people is not a language
    ("language", "language_family", "word_in", False),    # a family is not a word
    ("event", "person", "prohibited_in", True),           # nothing is "prohibited in" a person
])
def test_entity_types_are_not_conflated(center_type, node_type, predicate, subject_is_center):
    v = cs.check_relationship(center="Alpha", center_type=center_type, label="Beta", node_type=node_type, predicate=predicate,
                              subject_is_center=subject_is_center, relationship_text="", quote="Alpha Beta prohibited word speak", chunk_text="Alpha Beta prohibited word speak")
    assert v.status == "revised" and v.display == "is mentioned with" and "types do not fit" in v.reason


# --- 5. support ---------------------------------------------------------------------------------------------------------------

def test_a_typed_relationship_needs_a_cue_in_the_quote_or_it_is_weakened():
    v = cs.check_relationship(center="Inuit", center_type="people", label="Inuktitut", node_type="language", predicate="speaks", subject_is_center=True,
                              relationship_text="", quote=Q_INUIT, chunk_text=P11)
    assert v.status == "revised" and v.display == "is mentioned with" and "not stated in the quote" in v.reason     # "Inuktitut language" ≠ "speak"


def test_items_not_in_the_passage_or_not_together_with_the_centre_are_rejected():
    ghost = cs.check_relationship(center="Michif", center_type="language", label="Cree", node_type="language", predicate="part_of", subject_is_center=True,
                                  relationship_text="", quote=Q_METIS, chunk_text=P11)
    assert ghost.status == "rejected" and "not named" in ghost.reason
    apart = cs.check_relationship(center="Residential schools", center_type="institution", label="Jacques Cartier", node_type="person", predicate="led_by",
                                  subject_is_center=True, relationship_text="", quote="Cartier heard two captured guides", chunk_text="Cartier heard two captured guides speak")
    assert apart.status == "rejected" and "not near the quote" in apart.reason


def test_free_text_relationships_must_be_stated_by_the_quote():
    q = "The Canadarm was invented by Canadians in the 1970s."
    ok = cs.check_relationship(center="Canadarm", center_type="concept", label="Canadians", node_type="people", predicate="other", subject_is_center=True,
                               relationship_text="invented by", quote=q, chunk_text=q)
    assert ok.status == "ok" and ok.display == "invented by"
    bad = cs.check_relationship(center="Canadarm", center_type="concept", label="Canadians", node_type="people", predicate="other", subject_is_center=True,
                                relationship_text="include", quote=q, chunk_text=q)
    assert bad.status == "revised" and bad.display == "is mentioned with"


# --- 6. peoples are kept apart, and Aboriginal/Indigenous are not synonyms --------------------------------------------------------

def test_first_nations_metis_inuit_scopes_and_no_synonyms():
    assert cs.scopes_in("About 65% of the Aboriginal people are First Nations, while 30% are Métis and 4% Inuit") == {"First Nations", "Métis", "Inuit"}
    assert cs.scopes_in("Aboriginal peoples and Indigenous cultures") == set()           # umbrella words are not a scope and not a synonym
    # a centre about the Inuit accepts Inuktitut; a centre about the Métis accepts Michif; neither accepts the other
    assert cs.scope_conflict("Inuit languages", "Inuktitut", Q_INUIT, P11) == ""
    assert cs.scope_conflict("Métis languages", "Michif", Q_METIS, P11) == ""
    assert "Inuit" in cs.scope_conflict("Métis languages", "Inuktitut", Q_INUIT, P11)
    assert "Métis" in cs.scope_conflict("Inuit languages", "Michif", Q_METIS, P11)
    # the umbrella centre has no scope, so nothing is excluded by the guard
    assert cs.scope_conflict("Aboriginal languages", "Michif", Q_METIS, P11) == ""
    # an item whose passage names no group is not excluded from a First Nations centre
    assert cs.scope_conflict("First Nations cultures", "Creator", "relationship to the Creator, the natural environment", P10) == ""


# --- 7. old cached hubs still load; the prompt carries the rules ------------------------------------------------------------------

def test_hubs_saved_before_validation_still_load():
    old = {"central_concept": "Iroquoian", "kind": "concept", "summary": "s", "nodes": [
        {"id": "kanata", "label": "kanata", "relationship": "word in", "explanation": "x", "source_chunk_ids": ["c14"], "pages": ["14"],
         "evidence_quote": "the Iroquoian word kanata", "category": "concept"}], "summary_source_chunk_ids": [], "omitted": [], "central_category": "concept"}
    hub = ConceptHub.from_dict(old)
    assert isinstance(hub.nodes[0], ConceptNode) and hub.nodes[0].entity_type == "" and hub.nodes[0].relationship == "word in" and hub.central_entity_type == ""


def test_the_hub_prompt_states_the_type_direction_and_peoples_rules():
    from src import learning_extractors as ex
    p = ex._hub_prompt("First Nation Languages", "concept", list(SRC.values()), 6)
    for must in ("central_entity_type", "subject_is_center", "language_family", "a people is not a language", "mentioned_with", "three distinct peoples",
                 'never treat\n"Aboriginal" or "Indigenous" as the same', '"speaks":', '"word_in":'):
        assert must in p, must
    from src import config
    assert config.HUB_PROMPT_VERSION != config.LEARNING_PROMPT_VERSION                    # hubs re-generate; timelines keep their cache


# --- 8. weak links need the centre's topic near the quote; stated links come first ------------------------------------------------

def test_a_co_mention_needs_the_centres_topic_near_the_quote():
    reserve_q = "about half of First Nations people live on reserve land in about 600 communities"
    reserve_passage = "Aboriginal peoples today. In the 1970s, the term First Nations began to be used. " + reserve_q + ". Many communities are small."
    v = cs.check_relationship(center="First Nation Languages", center_type="language", label="Reserve communities", node_type="place", predicate="mentioned_with",
                              subject_is_center=True, relationship_text="", quote=reserve_q, chunk_text=reserve_passage)
    assert v.status == "rejected" and "'languages'" in v.reason and "not near the quote" in v.reason          # same passage and group word, but nothing about languages
    ok = cs.check_relationship(center="Aboriginal languages", center_type="language", label="Inuktitut", node_type="language", predicate="mentioned_with",
                               subject_is_center=True, relationship_text="", quote="which means “the people” in the Inuktitut language", chunk_text=P11)
    assert ok.status == "ok" and ok.support == "co-mentioned"                                  # "language" ~ "languages" is right there


def test_a_pronoun_quote_still_counts_through_the_previous_sentence():
    v = cs.check_relationship(center="Métis", center_type="people", label="Michif", node_type="language", predicate="mentioned_with", subject_is_center=True,
                              relationship_text="", quote=Q_METIS, chunk_text=P11)
    assert v.status == "ok"                                                                    # "They…" follows "The Métis are a distinct people…"


def test_stated_relationships_are_listed_before_co_mentions_and_survive_the_node_limit():
    raw = [node("Inuktitut", "language", "mentioned_with", Q_INUIT, "S1"),
           node("Residential schools", "institution", "prohibited_in", Q_SCHOOLS, "S3")]
    nodes, omitted = lv.validate_hub_nodes(raw, "Aboriginal languages", SRC, 1, "language")
    assert [n.label for n in nodes] == ["Residential schools"] and any("over the 1-node limit" in o for o in omitted)


def test_dialect_counts_as_the_language_topic_but_only_in_this_check():
    v = cs.check_relationship(center="Aboriginal languages", center_type="language", label="Michif", node_type="language", predicate="mentioned_with",
                              subject_is_center=True, relationship_text="", quote=Q_METIS, chunk_text=P11)
    assert v.status == "ok"                                                                    # "speak their own dialect, Michif"
    assert cs.scopes_in("Indigenous dialects of the Aboriginal peoples") == set()               # no synonym turns Aboriginal/Indigenous into a group


def test_a_sentence_that_lists_all_three_peoples_ties_an_item_to_none_of_them():
    tail = "About 65% of the Aboriginal people are First Nations, while 30% are Métis and 4% Inuit. the Inuktitut language, live in small, scattered communities."
    assert cs.nearest_scopes("Inuktitut", "the Inuktitut language, live in small", tail) == set()          # ambiguous: no tie either way
    assert cs.scope_conflict("First Nations languages", "Inuktitut", "the Inuktitut language, live in small", tail) == ""
    assert cs.nearest_scopes("Inuktitut", Q_INUIT, P11) == {"Inuit"}                                          # its own sentence names the Inuit
    assert cs.nearest_scopes("Michif", Q_METIS, P11) == {"Métis"}                                             # "They…" looks back to "The Métis are…"
