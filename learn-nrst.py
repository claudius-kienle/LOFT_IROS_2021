import argparse
from approaches import LOFT
import json
from pathlib import Path
from structs import Predicate, Literal, Type

from args import parse_args
from envs import create_env
from settings import create_config


def main(args):
    handover_dir = Path(args.out_dir)
    assert handover_dir.is_dir()
    action_preds = json.loads((handover_dir / "action-preds.json").read_text())
    preds = json.loads((handover_dir / "preds.json").read_text())
    episode = json.loads((handover_dir / "episode.json").read_text())

    action_predicates = {
        k: Predicate(
            name=v["name"],
            arity=v["arity"],
            var_types=[Type(t) for t in v["var_types"]],
            is_action_pred=True,
            sampler=(),
        )
        for k, v in action_preds.items()
    }

    pddl_preds = {
        k: Predicate(
            name=v["name"],
            arity=v["arity"],
            var_types=[Type(t) for t in v["var_types"]],
            holds=(),
        )
        for k, v in preds.items()
    }

    def parse_preds(ps):
        return [Literal(predicate=pddl_preds[p["predicate_name"]], variables=p["variables"]) for p in ps]

    def parse_action(a):
        predicate = action_predicates[action_predicates[a["action_pred_name"]]]
        return Literal(predicate=predicate, variables=a["variables"])

    p_episode = []
    for state, action, next_state, _ in episode:
        state = parse_preds(state)
        action = parse_action(action)
        next_state = parse_preds(next_state)

        data = (frozenset(state), action, frozenset(next_state))
        p_episode.append(data)

    state_preds = list(pddl_preds.values())
    action_preds = list(action_predicates.values())

    config = create_config(args)
    env = create_env(config)
    simulator = env.get_next_state
    approach = LOFT(config, simulator, state_preds, action_preds)
    approach.set_seed(0)
    print("Training approach...", flush=True)
    approach.train((p_episode, []))

    ops = "\n".join([operator.pddl_str() for operator in approach._operators])

    Path(handover_dir / "operators.txt").write_text(ops)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cover")
    parser.add_argument("--collect_data", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--out_dir", required=True, type=str)
    args = parser.parse_args()
    main(args)
