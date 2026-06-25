This repository contains agents (negotiators) submitted to the [ANL league](https://anac.cs.brown.edu/anl) of the [ANAC Competition](https://anac.cs.brown.edu)

To install this package just run:

```bash
pip install anl-agents
```

There are two ways to submit agents to this repository:

1. Participate in the ANAC competition [https://anac.cs.brown.edu/anl](https://anac.cs.brown.edu/anl)
2. Submit a pull-request with your agent added to the contrib directory.

# Getting lists of agents

You can get any specific subset of the agents in the library using `get_agents()`. This function
has the following parameters:

- version: Either a competition year (2024, ...) or the value "contrib" for all other agents. You can also pass "all" or "any" to get all agents.
- track: The track (advantage, utility, welfare, nash, kalai, kalai-smorodinsky)
- qualified_only: If true, only agents that were submitted to ANL and ran in the qualifications round will be
  returned.
- finalists_only: If true, only agents that were submitted to ANL and passed qualifications will be
  returned.
- winners_only: If true, only winners of ANL (the given version) will be returned.
- top_only: Either a fraction of finalists or the top n finalists with highest scores in the finals of
  ANL.
- as_class: If true, the agent classes will be returned otherwise their full class names.

For example, to get the top 10% of the "advantage" track finalists in year 2024 as strings, you can use:

```bash
get_agents(version=2024, track="advantage", finalists_only=True, top_only=0.1, as_class=False)
```


<!-- BEGIN generated standings region -->

<!-- BEGIN generated standings: 2026 -->

## ANAC 2026 Results

### Qualified agents (37)

| # | Agent | ID | Author | Team | Institute | Country |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AaNanteLucky | 21329 | Rinon Asanuma | Team 376 | Tokyo University of Agriculture and Technology | Japan |
| 2 | AdaptiveBathNegotiator | 21759 | 尘 | Team 425 | North China Electric Power University | China |
| 3 | Agent360 | 21043 | Noam Kazum | Team 360 | College of Management Academic Studies | Israel |
| 4 | AgentNexus_2.0 | 21639 | Nishanth Rajan | Team 407 | Leibniz Universität Hannover | Germany |
| 5 | Anchor | 21786 | David Izhaki | Team 512 | Bar-Ilan University | Israel |
| 6 | AOA007 | 21621 | Achiya Zigler | AOA007 | Bar-Ilan University | Israel |
| 7 | BadIronV4 | 21565 | Itamar Hassidim | Team 408 | Bar-Ilan University | Israel |
| 8 | BalanceAgent | 21688 | Tianzi Ma | CARC | Harbin Institute of Technology, Shenzhen | China |
| 9 | BetterCallAgentInfinityV1000 | 21762 | Steffan Wolter | BetterCallAgent | Leibniz Universität Hannover | Germany |
| 10 | ChangAgent | 21381 | Shengbo Chang | tnap | Tokyo University of Agriculture and Technology | Japan |
| 11 | CodexAgentAnl | 21802 | Ryota GENSEKI | Team 507 | Tokyo University of Agriculture and Technology | Japan |
| 12 | Cunning Merchant | 21589 | Kaan Gonenli | Team 485 | Özyeğin University | Türkiye |
| 13 | DecepTor | 21573 | MertAltintasOzu | Team 493 | Özyeğin University | Türkiye |
| 14 | EmEfAgent | 21044 | Emir ali Şahinoğulları | AgentInOzu | Özyeğin University | Türkiye |
| 15 | Erman Conceal Negotiator | 22139 | David Erman | Team 833 | Bar-Ilan University | Israel |
| 16 | GroupN | 21569 | cetinhosafci | Team 492 | Özyeğin University | Türkiye |
| 17 | HalucinatorAgent2026 | 21676 | Gayathri Rajeev | Hallucinators | Leibniz Universität Hannover | Germany |
| 18 | Ianos | 20963 | Konstantinos Katsaras | Team 340 | University of Macedonia | Greece |
| 19 | IscasAgent | 21124 | Yize Li | Team 391 | University of Chinese Academy of Sciences  Institute of Software,Chinese Academy of Sciences | China |
| 20 | LIonel | 21369 | wei | Team 423 | Nanjing University of Science and Technology | China |
| 21 | LoonyGryphon | 21353 | Harrison Oates | Team 418 | Australian National University | Australia |
| 22 | MajiKayo | 21584 | Kota Fujimoto | Team 344 | Aichi | Japan |
| 23 | MirageV145 | 22289 | Avinash Pathak | Team 422 | Independent | India |
| 24 | MiyaDreamBelief | 21328 | Miyashita Taisuke | Team 421 | Tokyo University of Agriculture and Technology | Japan |
| 25 | Nashty Negotiator 12 | 21795 | Lukas Knölker | Nashty Negotiators | Leibniz Universität Hannover | Germany |
| 26 | NegotiatorX_ANL | 21102 | Serhat Giydiren | TeamX | Özyeğin University | Türkiye |
| 27 | ohanl | 22257 | カズマ | チーム298 | Tokyo University of Agriculture and Technology | Japan |
| 28 | OverlapConceder | 20462 | Ali Jahani | Team 285 | University of Tehran | Iran |
| 29 | OzuNegotiator | 21413 | Umair Ahmad | Team 363 | Özyeğin University | Türkiye |
| 30 | PerikosV3 | 22271 | Kosuke Nakata | Team 412 | Tokyo University of Agriculture and Technology | Japan |
| 31 | Phantom8 | 21810 | Jeremy Hui | Team 358 | Rutgers University | United States |
| 32 | SBDANL | 22290 | Hajime Endo | Team Ukku | Tokyo University of Agriculture and Technology | Japan |
| 33 | Snake | 20829 | HIROTADA Matsumoto | Team 195 | Tokyo University of Agriculture and Technology | Japan |
| 34 | Staborn Negotiator | 22146 | Athina Georgara | Team 505 | University of Southampton | United Kingdom |
| 35 | tjqzagent | 21140 | TJQZ_Agent | Team 395 | Institute of Software, Chinese Academy of Sciences | China |
| 36 | WhaleV0.2 | 21350 | Omri Perry | Whales | Bar-Ilan University | Israel |
| 37 | XGAgent | 21728 | Fukutoku Yuma | Team 305 | Tokyo University of Agriculture and Technology | Japan |

Get them after install with:

```python
get_agents(2026, qualified_only=True)
```

**Disqualified (2):** phantom_etneg, TokyoV11

<!-- END generated standings: 2026 -->

<!-- BEGIN generated standings: 2025 -->

## ANAC 2025 Results

### Qualified agents (17)

| # | Agent | ID | Author | Team | Institute | Country |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | a4e | 20680 | カズマ | チーム298 | Tokyo University of Agriculture and Technology | Japan |
| 2 | Astrat3m | 20407 | Yunfei Wang | ChongqingAgent | Chongqing Jiaotong University | China |
| 3 | CARC2025 | 21892 | carc (anl2025) | carc (anl2025) | — | — |
| 4 | eoh_agent | 20711 | Hallel Weinberg | EOH | Bar-Ilan University | Israel |
| 5 | JeemNegotiator1 | 20485 | Mark Vexler | Team 291 | Bar-Ilan University | Israel |
| 6 | kAgent | 20744 | kon | Team 309 | colleagues | Japan |
| 7 | KDY | 20543 | Yuji Kobayashi | Tema KDY | Tokyo University of Agriculture and Technology | Japan |
| 8 | OzUAgent | 21893 | team_156 | team_156 | — | — |
| 9 | ProbaBot | 20424 | Loes Peters | Team 273 | Centrum Wiskunde & Informatica (CWI) | Netherlands |
| 10 | RivAgent | 20400 | Jumpei Kawahara | Natures | Tokyo University of Agriculture and Technology | Japan |
| 11 | RUFL | 20826 | Garrett Seo | Team 271 | Rutgers University | United States |
| 12 | SacAgent | 20805 | Hosssein Savari | University of Tehran | University of Tehran | Iran |
| 13 | Smart Negotiator | 20442 | Jonathan Mandl | Dream Team | Bar-Ilan University | Israel |
| 14 | StarGold15 | 20700 | Or Aharoni | Team 287 | Bar-Ilan University | Israel |
| 15 | TheMemorizer | 20715 | Ido | Team 307 | Bar-Ilan University | Israel |
| 16 | UfunATAgent | 21894 | team_305 | team_305 | — | — |
| 17 | Wagent | 20572 | Yunfei Wang | Team 300 | Chongqing Jiaotong University | China |

<!-- END generated standings: 2025 -->

<!-- BEGIN generated standings: 2024 -->

## ANAC 2024 Results

### Qualified agents (22)

| # | Agent | ID | Author | Team | Institute | Country |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Agent007 | 20359 | Talha Özer | Team 236 | Özyeğin University | Türkiye |
| 2 | AgentKB | 21891 | teamkb (anl2024) | teamkb (anl2024) | — | — |
| 3 | AgentNyan | 20161 | Ryota Arakawa | Team 123 | Tokyo University of Agriculture and Technology | Japan |
| 4 | AgentRenting2024 | 20245 | Mick Elshout | Team Renting | Utrecht University | Netherlands |
| 5 | AntiAgent | 20238 | Panagiotis Aronis | AntiAgents | Utrecht University | Netherlands |
| 6 | BargainBot | 20280 | Dmitrii Ratusniuc | Team 209 | Utrecht University | Netherlands |
| 7 | BidBot | 20365 | Azra Oymaagac | Team 240 | Özyeğin University | Türkiye |
| 8 | CARCAgent | 20344 | Tianzi Ma | CARC | Harbin Institute of Technology, Shenzhen | China |
| 9 | chaos-agent-anl | 20248 | Ayan Sengupta | AloneInTokyo | Independent | Japan |
| 10 | Goldie | 20282 | Ethan Beaird | Tulsa Eulers | University of Tulsa | United States |
| 11 | Group6 | 20357 | Asim Ozturk | Team 235 | Özyeğin University | Türkiye |
| 12 | INegotiator | 21889 | team_191 | team_191 | — | — |
| 13 | KatlaNirAgent | 20362 | Omer Bartfeld | TipsOnly | College of Management Academic Studies | Israel |
| 14 | KosAgent | 20269 | Kosuke Nakata | Team 205 | Tokyo University of Agriculture and Technology | Japan |
| 15 | MissG | 20226 | Bug Lee | Team 182 | Utrecht University | Netherlands |
| 16 | MyAgent86 | 20364 | Arda Kucuk | Team 234 | Özyeğin University | Türkiye |
| 17 | Nayesian2 | 21888 | susumu | susumu | — | — |
| 18 | Shochan | 20168 | takayama | takafam | Tokyo University of Agriculture and Technology | Japan |
| 19 | SliBwaTer | 20231 | Pieter van der Werff | Team Twistin' | Utrecht University | Netherlands |
| 20 | TAKAgent | 21890 | team_232 | team_232 | — | — |
| 21 | The Dealmaker | 20230 | Ilan Brilovitch | Team 186 | College of Management Academic Studies | Israel |
| 22 | UOAgent | 20277 | Hirotada Matsumoto | Team moto | Tokyo University of Agriculture and Technology | Japan |

<!-- END generated standings: 2024 -->

<!-- END generated standings region -->

# Winners of the ANL 2025 Competition

- First Place (tie): RUFL
- First Place (tie): SacAgent
- Third Place: UfunAtAgent

You can get these agents after installing anl-agents by running:

```bash
get_agents(2025, winners_only=True)
```


# Winners of the ANL 2024 Competition

## Advantage Track

- First Place: Shochan
- Second Place: UOAgent
- Third Place: AgentRenting2024

You can get these agents after installing anl-agents by running:

```bash
get_agents(2024, track="advantage", winners_only=True)
```

## Nash Track

- First Place: Shochan

You can get this agent after installing anl-agents by running:

```bash
get_agents(2024, track="nash", winners_only=True)
```

# Installation Note

If you are on Apple M1, you will need to install tensorflow **before** installing this package on conda using the method described [here](https://developer.apple.com/metal/tensorflow-plugin/)
