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
