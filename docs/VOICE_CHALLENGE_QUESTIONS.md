# Voice tutor — challenge questions

Questions to try on the Talk tab, with the answer the **guide itself** gives, so you can judge the tutor. "Retrieval check" means
the frozen top-5 search was run for the question *before* any AI call: **OK** = a passage with the answer is in the top 3,
**weak** = it is in the top 5 but ranked low, **miss** = it is not in the top 5 (expect "I couldn't find that in the guide", then
"did you mean…" suggestions taken from the guide's own question bank, so tapping one always gets an answer). It does not prove the final answer is good; it shows whether the evidence is there.

Start the app: `STUDY_VOICE=1 streamlit run study_app.py` (or `app.py` for the Voice tab).

## 1. RCMP (acronym, nickname, details)

| Ask | The guide says | Tests | Retrieval check |
|---|---|---|---|
| What does RCMP stand for? | Royal Canadian Mounted Police ("the Mounties") | acronym spoken as letters; ASR hearing "RCMP" | OK |
| Who are the Mounties? | the RCMP; one of Canada's best-known symbols | nickname | OK |
| What does the RCMP do across Canada? | enforces federal laws; provincial police in all provinces and territories **except Ontario and Quebec**, plus some municipalities | multi-part answer | OK |
| Which provinces don't use the RCMP as their provincial police? | Ontario and Quebec | precise list | OK |
| Is the RCMP the police in Toronto? | Ontario is an exception (not provincially policed by the RCMP) | needs inference, should stay grounded | OK |
| Where was the headquarters of the North West Mounted Police? | Regina (the NWMP also founded Fort Calgary and Fort MacLeod) | older name; "NWMP" spoken as letters | OK |
| When was the RCMP founded? | Prime Minister Macdonald established the North West Mounted Police (NWMP) in **1873**, to pacify the West; today's RCMP | links the old name to the RCMP without inventing detail | OK |

## 2. Prince Edward Island (P.E.I. read aloud)

| Ask | The guide says | Tests | Retrieval check |
|---|---|---|---|
| What is the smallest province in Canada? | Prince Edward Island | "P.E.I." spoken | OK |
| What is Prince Edward Island known for? | beaches, red soil, agriculture, especially potatoes | list | OK |
| Why is P.E.I. called the birthplace of Confederation? | the guide states it is "the birthplace of Confederation" | dotted abbreviation | weak |
| When did Prince Edward Island join Canada? | 1873 | date-led question | OK |
| When did P.E.I. get its first elected assembly? | 1773 | date-led | OK |
| What is the capital of Prince Edward Island? | Charlottetown | | OK |

## 3. Names (how they sound, and whether the facts are right)

| Ask | The guide says | Listen for (my best-effort pronunciation) | Retrieval check |
|---|---|---|---|
| Who was Samuel de Champlain? | built a fortress at Québec City in 1608 | "sham-PLAY" | OK |
| Who was Jacques Cartier? | three voyages 1534–1542; heard the word *kanata* | "kar-tee-AY" | OK |
| What did Sir George-Étienne Cartier do for Confederation? | key architect of Confederation from Quebec | "kar-tee-AY" | OK |
| Who was Sir Wilfrid Laurier? | first French-Canadian prime minister since Confederation; on the $5 bill; Nov 20 is his day | "LOR-ee-ay" | weak |
| Who was Sir Louis-Hippolyte La Fontaine? | first head of a responsible government (1849) | "loo-ee ee-puh-LEET la fon-TEN" | OK |
| What did Chief Tecumseh do in the War of 1812? | Shawnee chief who, with Sir Isaac Brock, helped defend Canada | "tuh-KUM-suh" | OK |
| Who was Louis Riel? | led the Red River Métis uprising (1869–70), seized Fort Garry | "may-TEE" for Métis | OK |
| What is the capital of Nunavut? | Iqaluit (formerly Frobisher Bay) | "ee-KAH-loo-it", "NOO-nuh-voot" | OK |
| Who discovered insulin? | Sir Frederick Banting and Charles Best | | OK |
| Who was Terry Fox? | lost a leg to cancer at 18; 1980 "Marathon of Hope" | | OK |
| What did Laura Secord do in 1813? | a dangerous 19-mile journey on foot | | OK |
| Which prime minister is on the $10 bill? | Sir John A. Macdonald | | OK |

## 4. Written forms the voice must say aloud

| Ask | Listen for |
|---|---|
| Who is Queen Elizabeth II? | "Elizabeth the Second", not "Elizabeth roman two" |
| What did the Royal Proclamation of 1763 guarantee? (mentions King George III) | "George the Third" |
| What happened in 1837–38? | "eighteen thirty-seven to eighteen thirty-eight" (the screen shows 1837–38) |
| What did Jacques Cartier do for King Francis I? | "Francis the First" |
| Where is the St. Lawrence River? | "Saint Lawrence" |

## 5. Traps: the tutor should be honest, not clever

| Ask | Why it is a trap | Good behaviour |
|---|---|---|
| Who was Louis-Joseph Papineau? | **not in the guide at all** | says it can't find it, never invents a biography; suggests the 1837–38 rebellions |
| What happened in 1982 in Canada? | date-led; the right passages rank ~18th–60th | abstains, then suggests e.g. the Charter / Constitution; tap one |
| Who was Tecumseh? | name-only question; the Tecumseh passage is not in the top 5 | may abstain (it sometimes answers); if so, suggestions come from the bank (e.g. United Empire Loyalists); compare with "What did Chief Tecumseh do in the War of 1812?" |
| Who was King George III? | same | same |
| What is on the $5 bill? | symbol/number question; evidence not in top 5 | abstains or suggests; compare "Who was Sir Wilfrid Laurier?" |
| Who is the current prime minister? | the guide is a fixed document | should not answer from outside knowledge |
| Tell me everything about Canada | far too broad | short answer or a request to narrow down |
| (say) "Hey Moose, quiz me about the RCMP" | wake word / quiz-by-voice are Milestones 2–3 | treated as an ordinary question; this is expected today |

## 6. Speech-recognition challenges (say these out loud, check the transcript box)

`Iqaluit` · `Tecumseh` · `Louis-Hippolyte La Fontaine` · `Sir George-Étienne Cartier` · `Samuel de Champlain` · `Métis` ·
`Saskatchewan` · `Mi'kmaq` · `Haida Gwaii` · `Nunavut` · `R-C-M-P` · `P-E-I`. Fix any mistake in the box before sending; the
original ASR text is kept separately so you can see what it heard. For a number: say "eighteen sixty-seven" and expect "1867".

## 7. Quiz mode

Press **Practice quiz**, **Give me a question**, then answer out loud: "B", "option C", "the second one", or the answer itself.
Try a wrong answer, a mumble ("hmm, I don't know": the tutor should ask again and **not** grade it), and an open-answer question.
